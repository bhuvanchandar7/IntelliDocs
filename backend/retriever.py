from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from backend.config import settings

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Reranker loaded.")

    def rerank(self, query: str, documents: list[Document], top_k=5) -> list[Document]:
        if not documents:
            return []
            
        # Create pairs [query, doc_text]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Associate scores with docs
        scored_docs = []
        for i, score in enumerate(scores):
            doc = documents[i]
            doc.metadata["score"] = float(score) # Add score to metadata
            scored_docs.append((doc, score))
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k docs
        print("Reranking Results:")
        for doc, score in scored_docs[:top_k]:
             print(f"  [{score:.4f}] {doc.metadata.get('source', 'unknown')}: {doc.page_content[:50]}...")
             
        return [doc for doc, score in scored_docs[:top_k]]

def get_vector_retriever(k=5):
    """
    Returns the Vector Store Retriever (Semantic Search).
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": k}), vectorstore

def get_bm25_retriever(documents, k=5):
    """
    Returns the BM25 Retriever (Keyword Search).
    Note: Requires the raw documents to initialize.
    """
    if not documents:
        return None
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


