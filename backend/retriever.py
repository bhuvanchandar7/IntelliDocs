from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from chromadb.config import Settings

def get_vector_retriever(k=5):
    """
    Returns the Vector Store Retriever (Semantic Search).
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    vectorstore = Chroma(
        collection_name="intellidocs_papers",
        embedding_function=embeddings,
        client_settings=Settings(allow_reset=True, anonymized_telemetry=False),
        host="localhost",
        port=8000
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(documents, k=5):
    """
    Returns the BM25 Retriever (Keyword Search).
    Note: Requires the raw documents to initialize.
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever

def get_hybrid_retriever(documents, k=5, alpha=0.5):
    """
    Returns an EnsembleRetriever combining Vector and BM25.
    alpha: Weight for Vector Search (0.5 means equal weight).
    """
    vector_retriever = get_vector_retriever(k=k)
    bm25_retriever = get_bm25_retriever(documents, k=k)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[alpha, 1 - alpha]
    )
    
    return ensemble_retriever
