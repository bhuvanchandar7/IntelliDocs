import os
import json
import random
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from openai import OpenAI

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in your environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))

def generate_question_answer(context, mock=False):
    """
    Generates a Q&A pair based on the context using GPT-3.5/4.
    """
    if mock:
        return {
            "instruction": "What is the main topic of this text?",
            "input": context[:100] + "...",
            "output": "This text discusses [Topic] based on the provided context."
        }

    prompt = f"""
    You are an expert at creating training data for LLMs.
    Given the following context from a research paper, generate a high-quality question and answer pair.
    
    Context:
    {context}
    
    Format your response as a JSON object with 'question' and 'answer' keys.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "instruction": data["question"],
            "input": "", # For RAG, context is usually retrieved, but for training we might want to feed it or just teach Q->A
            "output": data["answer"],
            "context": context # Keep context for reference
        }
    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return None

def main(output_file="data/qa_dataset.jsonl", limit=10, mock=True):
    # Connect to ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma(
        collection_name="intellidocs_papers",
        embedding_function=embeddings,
        client_settings=Settings(allow_reset=True, anonymized_telemetry=False),
        host="localhost",
        port=8000
    )

    # Fetch all IDs (or a subset) - Chroma allows getting by ID or generic get
    # For simplicity, we'll try to fetch a batch of random documents
    # In a real scenario, you'd iterate over all docs. 
    # Here we just get the first N items in the collection.
    
    print("Fetching documents from ChromaDB...")
    result = vectorstore.get(limit=limit, include=["documents", "metadatas"])
    documents = result["documents"]
    
    if not documents:
        print("No documents found in ChromaDB. Run ingestion first.")
        return

    print(f"Generating {len(documents)} Q&A pairs (Mock: {mock})...")
    
    with open(output_file, "w") as f:
        for i, doc_text in enumerate(documents):
            qa_pair = generate_question_answer(doc_text, mock=mock)
            if qa_pair:
                f.write(json.dumps(qa_pair) + "\n")
                if (i+1) % 5 == 0:
                    print(f"Generated {i+1} pairs...")

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--mock", action="store_true", help="Use mock generation (no API cost)")
    parser.add_argument("--output", type=str, default="data/qa_dataset.jsonl")
    args = parser.parse_args()
    
    main(args.output, args.limit, args.mock)
