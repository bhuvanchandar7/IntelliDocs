import time
import os
import warnings

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from backend.retriever import get_vector_retriever

def test_retrieval():
    print("Initializing Vector Retriever...")
    retriever = get_vector_retriever(k=3)
    
    query = "transformer attention mechanism"
    print(f"\nQuerying: '{query}'")
    
    start = time.time()
    docs = retriever.get_relevant_documents(query)
    end = time.time()
    
    print(f"Retrieved {len(docs)} documents in {end - start:.4f}s")
    
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}] {doc.metadata.get('source', 'Unknown')}")
        print(f"Content snippet: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test_retrieval()
