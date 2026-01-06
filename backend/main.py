import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import QueryRequest, QueryResponse, DocumentSource, HealthResponse
from backend.retriever import get_vector_retriever  # Currently using vector retriever
# from backend.retriever import get_hybrid_retriever # We'd need to load persistent docs for BM25

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

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="IntelliDocs API is running")

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        start_time = time.time()
        
        # 1. Retrieve Documents
        # Note: In a real/fine-tuned scenario, we would use 'get_hybrid_retriever'
        # but that requires re-indexing BM25 on startup. Keeping it simple with Vector.
        print(f"Querying: {request.query}")
        docs = retriever.get_relevant_documents(request.query)
        
        # 2. Generate Answer (Mock for now, or use LLM if configured)
        # TODO: Week 5-6 fine-tuned model would be invoked here.
        # For MVP Week 7, we'll return a comprehensive summary of the retrieved docs.
        
        # Simple extraction of context for the response
        generated_answer = "Based on the retrieved documents:\n\n"
        for i, doc in enumerate(docs):
            generated_answer += f"- {doc.page_content[:200]}...\n"
        
        generated_answer += "\n(Note: LLM Generation is the next component to integrate)"

        # 3. Format Response
        sources = [
            DocumentSource(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=0.0 # LC Chroma retriever doesn't always return scores easily in this interface
            ) for doc in docs
        ]
        
        process_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=generated_answer,
            sources=sources,
            query_time_ms=process_time
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
