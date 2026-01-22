import os
import logging
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import LlamaCpp
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
PDF_DIR = "data/raw_pdfs"
OUTPUT_FILE = "data/testset.json"

class SafeLlamaCpp(LlamaCpp):
    """Wrapper to fix list index out of range error in langchain agenerate."""
    async def agenerate(self, prompts, stop=None, callbacks=None, **kwargs):
        # Ragas 0.1.0 passes callbacks=[] which triggers an IndexError in Langchain
        if callbacks is not None and isinstance(callbacks, list) and len(callbacks) == 0:
            print("DEBUG: SafeLlamaCpp intercepted empty callbacks list. Fixing...", flush=True)
            callbacks = None
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks, **kwargs)

def main():
    import sys
    # Force flush stdout/stderr for Docker logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    try:
        logger.info("--- Starting Golden Dataset Generation (Debug Mode) ---")
        
        # 1. Load Documents
        logger.info(f"Loading PDFs from {PDF_DIR}...")
        loader = DirectoryLoader(PDF_DIR, glob="*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        if not documents:
            logger.error(f"No documents found in {PDF_DIR}. Please add PDFs first.")
            return
        logger.info(f"Loaded {len(documents)} pages.")

        # 2. Setup Local LLM (Generator & Critic)
        logger.info("Loading Local Mistral-7B...")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            return

        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        
        # Configure callbacks
        callbacks = [StreamingStdOutCallbackHandler()]

        llm = SafeLlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=0, # CPU only for stability in Docker
            temperature=0.7,
            verbose=True,
            callbacks=callbacks
        )

        # 3. Setup Embeddings
        logger.info("Loading BGE Embeddings...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 4. Initialize Generator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.testset.docstore import InMemoryDocumentStore
        from ragas.testset.extractor import KeyphraseExtractor
        
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_emb = LangchainEmbeddingsWrapper(embeddings)
        
        # Initialize Extractor (Required for DocStore to index content)
        extractor = KeyphraseExtractor(llm=ragas_llm)

        # Initialize DocStore containing the documents (converted/wrapped)
        docstore = InMemoryDocumentStore(
            splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128),
            embeddings=ragas_emb,
            extractor=extractor
        )

        generator = TestsetGenerator(
            generator_llm=ragas_llm,
            critic_llm=ragas_llm,
            embeddings=ragas_emb,
            docstore=docstore
        )

        # 5. Generate Testset
        logger.info("Generating Testset (Reduced to 3 samples)...")
        # Generating 3 samples to test the pipeline (reduced from 5)
        testset = generator.generate_with_langchain_docs(
            documents,
            test_size=3, 
            distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2}
        )

        # 6. Save Results
        logger.info(f"Saving testset to {OUTPUT_FILE}...")
        df = testset.to_pandas()
        df.to_json(OUTPUT_FILE, orient="records", indent=4)
        logger.info("Done! Preview of generated data:")
        print(df.head())
    
    except Exception as e:
        logger.exception("FATAL ERROR in dataset generation:")
        raise

if __name__ == "__main__":
    main()
