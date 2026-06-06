import time
import os
import shutil
import json
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

from backend.config import settings
from backend.llm import get_llm
from backend.schemas import QueryRequest, QueryResponse, DocumentSource, HealthResponse, MetricsResponse, DocumentListResponse, DocumentInfo
from backend.retriever import get_vector_retriever, get_bm25_retriever, Reranker
from ingestion.process_pdfs import ingest_single_file

# Configure logger
logger = logging.getLogger("intellidocs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global Components
vector_retriever = None
vectorstore = None
bm25_retriever = None
reranker = None
ensemble_retriever = None

# Global lock for updating the ensemble/BM25 retrievers
retriever_lock = threading.Lock()

def rebuild_ensemble_retriever():
    """
    Thread-safe helper to rebuild the BM25 and Ensemble retrievers
    whenever documents are ingested or deleted.
    """
    global bm25_retriever, ensemble_retriever, vector_retriever, vectorstore
    
    logger.info("Rebuilding Ensemble / BM25 Retriever...")
    with retriever_lock:
        try:
            if not vectorstore:
                logger.warning("Vectorstore is not initialized. Skipping BM25 rebuild.")
                ensemble_retriever = None
                return

            # Fetch all docs from Chroma
            data = vectorstore.get() 
            all_docs = []
            if data and data.get('documents'):
                 for i, text in enumerate(data['documents']):
                      meta = data['metadatas'][i] if data['metadatas'] else {}
                      all_docs.append(Document(page_content=text, metadata=meta))
            
            if all_docs:
                bm25_retriever = get_bm25_retriever(all_docs, k=10)
                # Create Ensemble Retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5] # Equal weight
                )
                logger.info(f"BM25 Index rebuilt successfully with {len(all_docs)} documents.")
            else:
                logger.warning("No documents found in Vectorstore. BM25 skipped.")
                bm25_retriever = None
                ensemble_retriever = vector_retriever # Fallback directly to vector search
        except Exception as e:
            logger.error(f"Error rebuilding BM25: {e}", exc_info=True)
            bm25_retriever = None
            ensemble_retriever = vector_retriever # Fallback

# Lifespan manager to handle startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_retriever, vectorstore, bm25_retriever, reranker, ensemble_retriever
    
    logger.info("Initializing Application...")
    
    # 1. Initialize Vector Store & Retriever
    logger.info("Initializing Vector Retriever...")
    vector_retriever, vectorstore = get_vector_retriever(k=10) # Fetch more candidates
    
    # 2. Build BM25 (Requires fetching all docs)
    rebuild_ensemble_retriever()
    
    # 3. Initialize Reranker
    logger.info("Loading Reranker Model...")
    reranker = Reranker()

    # 4. Load LLM
    try:
        logger.info("Initializing LLM Model...")
        app.state.llm = get_llm()
        logger.info("LLM Loaded Successfully.")
    except Exception as e:
        logger.error(f"Warning: Failed to load LLM. Queries may fail: {e}")
        app.state.llm = None
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="IntelliDocs API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Stats Tracker
stats = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "timestamps": [] 
}

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        # Get count
        count = vectorstore._collection.count() if vectorstore else 0
        
        avg_latency = 0.0
        if stats["total_requests"] > 0:
            avg_latency = stats["total_latency_ms"] / stats["total_requests"]
        
        # Calculate Request History
        now = time.time()
        one_hour_ago = now - 3600
        stats["timestamps"] = [ts for ts in stats["timestamps"] if ts > one_hour_ago]
        history = [0] * 60
        for ts in stats["timestamps"]:
            minute_idx = int((ts - one_hour_ago) / 60)
            if 0 <= minute_idx < 60:
                history[minute_idx] += 1
            
        return MetricsResponse(
            document_count=count,
            total_requests=stats["total_requests"],
            average_latency_ms=round(avg_latency, 2),
            vector_db_status="Healthy",
            request_history=history
        )
    except Exception as e:
        return MetricsResponse(
            document_count=0,
            total_requests=stats["total_requests"],
            average_latency_ms=0.0,
            vector_db_status=f"Error: {str(e)}",
            request_history=[0]*60
        )

@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    if not vectorstore:
         raise HTTPException(status_code=503, detail="Vectorstore not initialized")
         
    data = vectorstore.get(limit=10000, include=["metadatas"])
    docs = []
    ids = data["ids"]
    metadatas = data["metadatas"]
    
    for i, doc_id in enumerate(ids):
        docs.append(DocumentInfo(
            id=doc_id,
            source=metadatas[i].get("source", "Unknown"),
            page_content="",
            metadata=metadatas[i] or {}
        ))
    
    return DocumentListResponse(documents=docs)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="IntelliDocs API is running")

@app.post("/query") 
async def query_documents(request: QueryRequest):
    try:
        start_time = time.time()
        stats["total_requests"] += 1
        stats["timestamps"].append(time.time())
        
        logger.info(f"Querying: {request.query}")
        
        if not ensemble_retriever:
            raise HTTPException(status_code=503, detail="Retrieval system is not initialized.")
            
        # 1. Retrieval (Hybrid)
        candidates = ensemble_retriever.invoke(request.query)
        logger.info(f"Retrieved {len(candidates)} candidates.")
        
        # 2. Reranking
        docs = reranker.rerank(request.query, candidates, top_k=5)
        logger.info(f"After Reranking: Keeping {len(docs)} best docs.")

        # Prepare Context
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        if not hasattr(app.state, "llm") or app.state.llm is None:
             raise HTTPException(status_code=503, detail="LLM model is not loaded.")
            
        prompt = f"<s>[INST] You are a helpful assistant. Answer the question based strictly on the context below.\n\nContext:\n{context_text}\n\nQuestion: {request.query} [/INST]"
        
        async def generate_stream():
            sources = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("score", 0.0)
                } for doc in docs
            ]
            
            yield json.dumps({"type": "sources", "data": sources}) + "\n"
            
            try:
                for chunk in app.state.llm.stream(prompt):
                     # Handle ChatModel chunks (AIMessageChunk) vs standard text strings
                     text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                     if text:
                        yield json.dumps({"type": "token", "data": text}) + "\n"
            except Exception as stream_err:
                logger.error(f"Error during streaming: {stream_err}")
                yield json.dumps({"type": "token", "data": f"\n\n[Error during LLM generation: {stream_err}]"}) + "\n"
            
            process_time = (time.time() - start_time) * 1000
            stats["total_latency_ms"] += process_time
            yield json.dumps({"type": "done", "latency_ms": process_time}) + "\n"

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def background_ingest(file_path, vectorstore_arg):
    try:
        logger.info(f"Starting background ingestion for {file_path}...")
        count = ingest_single_file(file_path, vectorstore)
        
        # Rebuild retrievers after new file is ingested
        rebuild_ensemble_retriever()
        
        logger.info(f"Background ingestion complete. {count} chunks added.")
    except Exception as e:
        logger.error(f"Background ingestion failed for {file_path}: {e}", exc_info=True)

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        save_dir = "data/raw_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        background_tasks.add_task(background_ingest, file_path, vectorstore)
        
        return {"message": f"Upload accepted. Processing {file.filename} in background.", "status": "processing"}
            
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
def delete_document(source: str = Query(..., description="The source filename to delete (e.g. report.pdf)")):
    try:
        if not vectorstore:
             raise HTTPException(status_code=503, detail="Vectorstore not initialized")
             
        logger.info(f"Deleting documents with source: {source}")
        vectorstore._collection.delete(where={"source": source})
        
        file_path = os.path.join("data/raw_pdfs", source)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Rebuild retrievers after deletion
        rebuild_ensemble_retriever()
            
        return {"message": f"Successfully deleted documents for {source}"}
    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)


