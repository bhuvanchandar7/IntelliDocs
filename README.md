# IntelliDocs

Intelligent Document Retrieval System (RAG).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Infrastructure**:
    ```bash
    docker-compose up -d
    ```

## Data Ingestion

1.  **Fetch Papers**:
    ```bash
    python ingestion/fetch_papers.py --limit 5
    ```

2.  **Process and Chunk**:
    ```bash
    python ingestion/process_pdfs.py
    ```

## Project Structure
- `ingestion/`: Scripts for fetching and processing data.
- `data/`: Local storage for PDFs and VectorDB.
- `backend/`: FastAPI application.

## Fine-Tuning (Weeks 5-6)

The project includes full scaffolding for fine-tuning Mistral-7B.

### Prerequisites (for full execution)
1.  **OpenAI API Key**: Required for generating synthetic training data.
2.  **GPU (NVIDIA)**: Required for running `train.py` with 4-bit quantization (QLoRA). Mac/CPU is not supported for efficient training.

### Steps
1.  **Generate Data**:
    ```bash
    export OPENAI_API_KEY=sk-...
    python finetuning/generate_qa.py --limit 500  # Remove --mock for real data
    ```
2.  **Train Model**:
    ```bash
    python finetuning/train.py --data data/qa_dataset.jsonl
    ```