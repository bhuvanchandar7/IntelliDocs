import time
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import QueryRequest, QueryResponse, DocumentSource, HealthResponse, MetricsResponse, DocumentListResponse, DocumentInfo
from backend.retriever import get_vector_retriever, get_bm25_retriever, Reranker
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from ingestion.process_pdfs import ingest_single_file

from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
import json

# Global Components
vector_retriever = None
vectorstore = None
bm25_retriever = None
reranker = None
ensemble_retriever = None

# Lifespan manager to handle startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_retriever, vectorstore, bm25_retriever, reranker, ensemble_retriever
    
    print("Initializing Application...")
    
    # 1. Initialize Vector Store & Retriever
    print("Initializing Vector Retriever...")
    vector_retriever, vectorstore = get_vector_retriever(k=10) # Fetch more candidates
    
    # 2. Initialize BM25 (Requires fetching all docs)
    print("Building BM25 Index (Hybrid Search)...")
    try:
        # Fetch all docs from Chroma
        data = vectorstore.get() 
        all_docs = []
        if data['documents']:
             for i, text in enumerate(data['documents']):
                  meta = data['metadatas'][i] if data['metadatas'] else {}
                  all_docs.append(Document(page_content=text, metadata=meta))
        
        if all_docs:
            bm25_retriever = get_bm25_retriever(all_docs, k=10)
            print(f"BM25 Index built with {len(all_docs)} documents.")
            
            # Create Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5] # Equal weight
            )
        else:
            print("Warning: No documents found. BM25 skipped.")
            ensemble_retriever = vector_retriever # Fallback
            
    except Exception as e:
        print(f"Error building BM25: {e}")
        ensemble_retriever = vector_retriever # Fallback

    # 3. Initialize Reranker
    print("Loading Reranker Model...")
    reranker = Reranker()

    # 4. Load LLM
    model_path = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if os.path.exists(model_path):
        print("Loading LLM model into memory...")
        from langchain_community.llms import LlamaCpp
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        app.state.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=4096,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=True,
            temperature=0.7,
        )
        print("LLM Loaded Successfully.")
    else:
        print("Warning: LLM model not found. Queries will fail.")
        app.state.llm = None
    
    yield
    
    print("Shutting down...")

app = FastAPI(title="IntelliDocs API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        
        print(f"Querying: {request.query}")
        
        # 1. Retrieval (Hybrid)
        # Fetch top 10 candidates
        candidates = ensemble_retriever.invoke(request.query)
        print(f"Retrieved {len(candidates)} candidates.")
        
        # 2. Reranking
        # Filter top 5
        docs = reranker.rerank(request.query, candidates, top_k=5)
        print(f"After Reranking: Keeping {len(docs)} best docs.")

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
                    # Score might be useful for debug but user doesn't need it
                    "score": doc.metadata.get("score", 0.0)
                } for doc in docs
            ]
            
            yield json.dumps({"type": "sources", "data": sources}) + "\n"
            
            for chunk in app.state.llm.stream(prompt):
                 if chunk:
                    yield json.dumps({"type": "token", "data": chunk}) + "\n"
            
            process_time = (time.time() - start_time) * 1000
            stats["total_latency_ms"] += process_time
            yield json.dumps({"type": "done", "latency_ms": process_time}) + "\n"

        return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def background_ingest(file_path, vectorstore_arg): # vectorstore_arg ignored, using global
    try:
        print(f"Starting background ingestion for {file_path}...")
        # Ignore vectorstore_arg, use global variable
        count = ingest_single_file(file_path, vectorstore)
        
        # TODO: Update BM25 index? 
        # For now, it won't update until restart. That's a trade-off.
        
        print(f"Background ingestion complete. {count} chunks added.")
    except Exception as e:
        print(f"Background ingestion failed for {file_path}: {e}")

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        save_dir = "data/raw_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Passing None for vectorstore, function will access global or create new
        # Correct approach: Pass the global vectorstore
        background_tasks.add_task(background_ingest, file_path, vectorstore)
        
        return {"message": f"Upload accepted. Processing {file.filename} in background.", "status": "processing"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
def delete_document(source: str = Query(..., description="The source filename to delete (e.g. report.pdf)")):
    try:
        if not vectorstore:
             raise HTTPException(status_code=503, detail="Vectorstore not initialized")
             
        print(f"Deleting documents with source: {source}")
        vectorstore._collection.delete(where={"source": source})
        
        file_path = os.path.join("data/raw_pdfs", source)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {"message": f"Successfully deleted documents for {source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

