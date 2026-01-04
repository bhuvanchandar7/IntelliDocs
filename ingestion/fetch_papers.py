import arxiv
import os
import argparse

def fetch_papers(query="cat:cs.CL", max_results=10, data_dir="data/raw_pdfs"):
    """
    Fetches papers from ArXiv and saves them as PDFs.
    
    Args:
        query (str): ArXiv query (default: cs.CL for Computation and Language).
        max_results (int): Number of papers to fetch.
        data_dir (str): Directory to save PDFs.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Fetching {max_results} papers for query: {query}...")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    downloaded_count = 0
    for result in client.results(search):
        try:
            filename = f"{result.entry_id.split('/')[-1]}.pdf"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading: {result.title}")
                result.download_pdf(dirpath=data_dir, filename=filename)
                downloaded_count += 1
            else:
                print(f"Skipping (already exists): {result.title}")
        except Exception as e:
            print(f"Error downloading {result.title}: {e}")

    print(f"Done. Downloaded {downloaded_count} new papers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from ArXiv.")
    parser.add_argument("--query", type=str, default="cat:cs.CL", help="ArXiv query category (e.g., cs.CL, cs.LG)")
    parser.add_argument("--limit", type=int, default=10, help="Number of papers to download")
    parser.add_argument("--output", type=str, default="data/raw_pdfs", help="Output directory")

    args = parser.parse_args()
    fetch_papers(args.query, args.limit, args.output)
