import time
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import QueryRequest, QueryResponse, DocumentSource, HealthResponse, MetricsResponse, DocumentListResponse, DocumentInfo
from backend.retriever import get_vector_retriever
from ingestion.process_pdfs import ingest_single_file

app = FastAPI(title="IntelliDocs API", version="0.1.0")

# CORS (Allow Frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Retriever (Lazy loading could be better but this is simple)
# Note: For efficient Hybrid Search we need to keep the BM25 index in memory.
# For now, we'll default to Vector Search (Chroma) which is persistent and efficient.
retriever = get_vector_retriever(k=3)


# Global Stats Tracker
stats = {
    "total_requests": 0,
    "total_latency_ms": 0.0
}

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        # Get count from Chroma
        count = retriever.vectorstore._collection.count()
        
        avg_latency = 0.0
        if stats["total_requests"] > 0:
            avg_latency = stats["total_latency_ms"] / stats["total_requests"]
            
        return MetricsResponse(
            document_count=count,
            total_requests=stats["total_requests"],
            average_latency_ms=round(avg_latency, 2),
            vector_db_status="Healthy"
        )
    except Exception as e:
        # Fallback if Chroma fails
        return MetricsResponse(
            document_count=0,
            total_requests=stats["total_requests"],
            average_latency_ms=0.0,
            vector_db_status=f"Error: {str(e)}"
        )

@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    # Fetch ALL documents (limit=None doesn't always work as expected in some chroma versions, setting high number)
    data = retriever.vectorstore.get(limit=10000, include=["metadatas"])
    
    docs = []
    ids = data["ids"]
    metadatas = data["metadatas"]
    # documents = data["documents"] # Not fetching content anymore
    
    # Debug logging
    unique_sources = set()
    for m in metadatas:
        if m and "source" in m:
            unique_sources.add(m["source"])
    print(f"DEBUG: list_documents found {len(ids)} chunks from sources: {unique_sources}")

    for i, doc_id in enumerate(ids):
        docs.append(DocumentInfo(
            id=doc_id,
            source=metadatas[i].get("source", "Unknown"),
            page_content="", # Preview not needed for list view, saves bandwidth
            metadata=metadatas[i] or {}
        ))
    
    return DocumentListResponse(documents=docs)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="IntelliDocs API is running")

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        start_time = time.time()
        
        # Update Stats
        stats["total_requests"] += 1
        
        # 1. Retrieve Documents
        print(f"Querying: {request.query}")
        docs = retriever.get_relevant_documents(request.query)
        
        # 2. Generate Answer with Local LLM
        print("Starting LLM Generation...")
        
        # Prepare Context
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Initializing LLM logic inside the request to handle model loading checks
        # In production, this should be global, but we need to check if model exists first
        model_path = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
             generated_answer = "Error: Model file not found. It might still be downloading. Please wait."
        else:
            # Lazy Import and Load (Global var would be better)
            from langchain_community.llms import LlamaCpp
            from langchain.callbacks.manager import CallbackManager
            from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

            # Check if global 'llm' exists (hacky singleton for this function scope)
            # A better pattern is to load it at startup, but the file might not exist yet.
            if not hasattr(app.state, "llm"):
                print("Loading LLM model into memory...")
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                app.state.llm = LlamaCpp(
                    model_path=model_path,
                    n_gpu_layers=-1, # Offload all to Metal/GPU
                    n_batch=512,
                    n_ctx=4096, # Context window
                    f16_kv=True,  # MUST set to True for Metal!
                    callback_manager=callback_manager,
                    verbose=True,
                    temperature=0.7,
                )
            
            # Mistral Instruct Prompt Format
            prompt = f"<s>[INST] You are a helpful assistant. Answer the question based strictly on the context below.\n\nContext:\n{context_text}\n\nQuestion: {request.query} [/INST]"
            
            # Generate
            # Using invoke for synchronous wait, stream is better for UX but requires event stream
            generated_answer = app.state.llm.invoke(prompt)

        # 3. Format Response
        sources = [
            DocumentSource(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=0.0
            ) for doc in docs
        ]
        
        process_time = (time.time() - start_time) * 1000
        stats["total_latency_ms"] += process_time
        
        return QueryResponse(
            answer=generated_answer,
            sources=sources,
            query_time_ms=process_time
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Wrapper for background task to handle errors logs
def background_ingest(file_path, vectorstore):
    try:
        print(f"Starting background ingestion for {file_path}...")
        count = ingest_single_file(file_path, vectorstore)
        print(f"Background ingestion complete. {count} chunks added.")
    except Exception as e:
        print(f"Background ingestion failed for {file_path}: {e}")

from fastapi import BackgroundTasks

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Ensure data directory exists
        save_dir = "data/raw_pdfs"
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, file.filename)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger Background Ingestion
        # Pass the global vectorstore to ensure we write to the active connection
        background_tasks.add_task(background_ingest, file_path, retriever.vectorstore)
        
        return {"message": f"Upload accepted. Processing {file.filename} in background.", "status": "processing"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
def delete_document(source: str = Query(..., description="The source filename to delete (e.g. report.pdf)")):
    try:
        # Delete from Chroma
        # vectorstore.delete expects ids or filter
        # We delete where metadata['source'] == source
        print(f"Deleting documents with source: {source}")
        
        # ChromaDB delete collection call
        retriever.vectorstore._collection.delete(where={"source": source})
        
        # Optional: Delete actual file from disk
        file_path = os.path.join("data/raw_pdfs", source)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {"message": f"Successfully deleted documents for {source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

