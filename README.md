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