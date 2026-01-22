import logging
import pandas as pd
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TESTSET_PATH = "data/testset.json"
RESULTS_PATH = "data/evaluation_results.csv"
API_URL = "http://localhost:8000/query" # Docker internal networking or localhost depending on where this runs
MODEL_PATH = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def get_answer_from_pipeline(question: str):
    """Query the actually running RAG pipeline (Handles Streaming Response)"""
    import json
    try:
        # Fix 1: Use 'query' instead of 'question'
        with requests.post(API_URL, json={"query": question}, stream=True) as response:
            response.raise_for_status()
            
            answer = ""
            contexts = []
            
            # Fix 2: Parse NDJSON stream
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data['type'] == 'sources':
                            # Fix 3: Use 'content' key
                            contexts = [item['content'] for item in data['data']]
                        elif data['type'] == 'token':
                            answer += data['data']
                    except json.JSONDecodeError:
                        continue
                        
            return {
                "answer": answer,
                "contexts": contexts
            }
    except Exception as e:
        logger.error(f"Error querying pipeline: {e}")
        return {"answer": "Error", "contexts": []}

class SafeLlamaCpp(LlamaCpp):
    """Wrapper to fix list index out of range error in langchain agenerate."""
    async def agenerate(self, prompts, stop=None, callbacks=None, **kwargs):
        if callbacks is not None and isinstance(callbacks, list) and len(callbacks) == 0:
            callbacks = None
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks, **kwargs)

def main():
    logger.info("--- Starting RAG Pipeline Evaluation ---")

    # 1. Load Testset
    logger.info(f"Loading testset from {TESTSET_PATH}...")
    try:
        df = pd.read_json(TESTSET_PATH, orient="records")
    except Exception as e:
        logger.error(f"Failed to load testset: {e}")
        return

    # 2. Collect Responses from Pipeline
    logger.info("Querying RAG Pipeline for answers...")
    answers = []
    contexts = []
    
    # We rename columns to match Ragas expectations
    # Ragas expects: question, answer, contexts, ground_truth
    # Our testset has: question, ground_truth, etc.
    
    for _, row in df.iterrows():
        result = get_answer_from_pipeline(row['question'])
        answers.append(result['answer'])
        contexts.append(result['contexts'])
        
    df['answer'] = answers
    df['contexts'] = contexts
    
    # Rename ground_truth column to ground_truth (it might be 'ground_truth_context' or similar depending on generation)
    # The generated testset usually has 'ground_truth'
    
    dataset = Dataset.from_pandas(df)

    # 3. Setup Metrics and Judges
    logger.info("Initializing Judge Models (Mistral-7B)...")
    
    # Judge LLM
    llm = SafeLlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=0,
        temperature=0.0, # Zero temp for evaluation consistency
        verbose=True
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # Judge Embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    # 4. Run Evaluation
    logger.info("Running Ragas Evaluation (Faithfulness, Relevance, Recall, Precision)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    # 5. Save Results
    logger.info(f"Saving results to {RESULTS_PATH}...")
    df_results = results.to_pandas()
    df_results.to_csv(RESULTS_PATH, index=False)
    
    print("\n\n=== Evaluation Summary ===")
    print(results)
    print("==========================\n")

if __name__ == "__main__":
    main()
