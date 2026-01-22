import logging
import pandas as pd
import requests
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TESTSET_PATH = "data/testset.json"
PREDICTIONS_PATH = "data/prediction_results.json"
API_URL = "http://localhost:8000/query"

def get_answer_from_pipeline(question: str):
    """Query the actually running RAG pipeline"""
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
            
            return {
                "question": question,
                "answer": answer.strip(),
                "contexts": contexts
            }
            
    except Exception as e:
        logger.error(f"Error querying pipeline: {e}")
        return {
            "question": question,
            "answer": "Error processing request", 
            "contexts": [],
            "error": str(e)
        }

def main():
    logger.info("--- RAG Pipeline Prediction Generator ---")

    # 1. Load Testset
    if not os.path.exists(TESTSET_PATH):
        logger.error(f"Testset not found at {TESTSET_PATH}")
        return
        
    logger.info(f"Loading testset from {TESTSET_PATH}...")
    try:
        df = pd.read_json(TESTSET_PATH, orient="records")
    except Exception as e:
        logger.error(f"Failed to load testset: {e}")
        return

    # 2. Process sequentially
    results = []
    
    logger.info(f"Generating answers for {len(df)} questions...")
    for index, row in df.iterrows():
        question = row['question']
        ground_truth = row.get('ground_truth', '')
        
        logger.info(f"Processing ({index+1}/{len(df)}): {question}")
        
        result = get_answer_from_pipeline(question)
        result['ground_truth'] = ground_truth
        results.append(result)

    # 3. Save Results
    logger.info(f"Saving predictions to {PREDICTIONS_PATH}...")
    with open(PREDICTIONS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n\n=== Prediction Summary ===")
    for res in results:
        print(f"\nQ: {res['question']}")
        print(f"A: {res['answer'][:100]}...") # Truncate for display
        print(f"Contexts Found: {len(res['contexts'])}")
    print("==========================\n")

if __name__ == "__main__":
    main()
