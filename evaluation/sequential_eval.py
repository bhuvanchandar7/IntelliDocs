import logging
import pandas as pd
import requests
import json
import time
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
API_URL = "http://localhost:8000/query"
MODEL_PATH = "data/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def get_answer_from_pipeline(question: str):
    """Query the actually running RAG pipeline (Handles Streaming Response)"""
    try:
        with requests.post(API_URL, json={"query": question}, stream=True) as response:
            response.raise_for_status()
            
            answer = ""
            contexts = []
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data['type'] == 'sources':
                            contexts = [item['content'] for item in data['data']]
                        elif data['type'] == 'token':
                            answer += data['data']
                    except json.JSONDecodeError:
                        continue
            
            if not contexts:
                contexts = ["No context retrieved"]

            return {
                "answer": answer.strip(),
                "contexts": contexts
            }
            
    except Exception as e:
        logger.error(f"Error querying pipeline: {e}")
        return {
            "answer": "Error processing request", 
            "contexts": ["Error retrieving context"]
        }

class SafeLlamaCpp(LlamaCpp):
    """Wrapper to fix list index out of range error in langchain agenerate."""
    async def agenerate(self, prompts, stop=None, callbacks=None, **kwargs):
        if callbacks is not None and isinstance(callbacks, list) and len(callbacks) == 0:
            callbacks = None
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks, **kwargs)

def main():
    logger.info("--- Starting Sequential RAG Pipeline Evaluation ---")

    # 1. Load Testset
    logger.info(f"Loading testset from {TESTSET_PATH}...")
    try:
        df = pd.read_json(TESTSET_PATH, orient="records")
    except Exception as e:
        logger.error(f"Failed to load testset: {e}")
        return

    # 2. Initialize Judge Models ONCE
    logger.info("Initializing Judge Models (Mistral-7B)...")
    llm = SafeLlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=0,
        n_batch=512,       # Reverted to standard 512 for stability
        n_threads=4,       # Low thread count
        f16_kv=True,       # Memory optimization
        temperature=0.0,   
        verbose=False      # Less noise
    )
    ragas_llm = LangchainLLMWrapper(llm)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    # 3. Process sequentially
    logger.info("Processing queries sequentially...")
    all_results = []
    
    for index, row in df.iterrows():
        question = row['question']
        ground_truth = row.get('ground_truth', '')
        logger.info(f"Processing ({index+1}/{len(df)}): {question}")

        # Validation: Check if we have ground truth, otherwise context_recall/precision might fail
        # But for now we proceed.
        
        # A. Get Answer
        result = get_answer_from_pipeline(question)
        
        # B. Prepare Single-Row Dataset
        single_data = {
            'question': [question],
            'contexts': [result['contexts']],
            'answer': [result['answer']],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(single_data)
        
        # C. Evaluate Single Row
        try:
            scores = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                ],
                llm=ragas_llm,
                embeddings=ragas_emb,
                raise_exceptions=False # Don't crash on one metric failure
            )
            
            # Merge results
            row_result = {
                'question': question,
                'answer': result['answer'],
                'contexts': result['contexts'],
                'ground_truth': ground_truth,
                **scores
            }
            all_results.append(row_result)
            logger.info(f"Scores: {scores}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate row {index}: {e}")
            # Add partial result
            all_results.append({
                'question': question,
                'answer': result['answer'],
                'contexts': result['contexts'],
                'error': str(e)
            })

    # 4. Save Final Results
    if all_results:
        logger.info(f"Saving results to {RESULTS_PATH}...")
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_PATH, index=False)
        print("\n\n=== Evaluation Summary ===")
        print(results_df.describe())
        print("==========================\n")
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()
