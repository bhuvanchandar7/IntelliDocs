import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.docstore.document import Document

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_pdfs(input_dir="data/raw_pdfs", chunk_size=512, chunk_overlap=50):
    """
    Reads PDFs, extracts text, and chunks it.
    """
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDFs in {input_dir}")

    total_chunks = 0
    all_chunks = []

    for file in files:
        pdf_path = os.path.join(input_dir, file)
        try:
            print(f"Processing {file}...")
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = text_splitter.split_text(raw_text)
            
            # Convert to Document objects with metadata
            docs = [Document(page_content=chunk, metadata={"source": file}) for chunk in chunks]
            all_chunks.extend(docs)
            
            print(f"  -> Extracted {len(chunks)} chunks.")
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"Total chunks generated: {total_chunks}")
    
    if total_chunks > 0:
        print("Generating embeddings and storing in ChromaDB...")
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        # Initialize Vector Store (Chroma)
        # We use the HTTP client to connect to the docker container
        vectorstore = Chroma(
            collection_name="intellidocs_papers",
            embedding_function=embeddings,
            persist_directory="data/chroma_db"
        )
        
        # Add documents in batches to avoid hitting limits or memory issues
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            print(f"  Inserting batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}...")
            vectorstore.add_documents(documents=batch)
            
        print("✓ Successfully ingested all chunks into ChromaDB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs and chunk text.")
    parser.add_argument("--input", type=str, default="data/raw_pdfs", help="Input directory")
    
    args = parser.parse_args()
    process_pdfs(args.input)
