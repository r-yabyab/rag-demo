import json
import re
from pathlib import Path
from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import chromadb
import fitz  # PyMuPDF

def extract_year_from_text(text):
    """
    Extract year from text patterns like 'accepted 16 March 2006'
    Returns the year as string or None if not found
    """
    patterns = [
        r'accepted\s+\d{1,2}\s+\w+\s+(\d{4})',  # accepted 16 March 2006
        r'received\s+\d{1,2}\s+\w+\s+(\d{4})',  # received 10 January 2006
        r'accepted\s+\w+\s+\d{1,2},?\s+(\d{4})', # accepted March 16, 2006
        r'received\s+\w+\s+\d{1,2},?\s+(\d{4})', # received January 10, 2006
        r'published\s+online\s+\d{1,2}\s+\w+\s+(\d{4})', # published online 20 March 2006
        r'available\s+online\s+\d{1,2}\s+\w+\s+(\d{4})', # available online 25 March 2006
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        return year_match.group(0)
    
    return None

def extract_title_from_text(text):
    """
    Extract paper title using PDF metadata and simple text patterns
    """
    # Strategy: Join all words and look for patterns after indicators
    clean_text = ' '.join(text.split())
    
    # Look for patterns after "Short communication" or "Research report"
    patterns = [
        r'Short communication\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
        r'Research report\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
        r'Brief report\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
        r'Review article\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\s+', ' ', title)
            if 10 < len(title) < 300:
                return title
    
    return None

def extract_authors_from_text(text):
    """
    Extract authors using simple pattern matching
    """
    clean_text = ' '.join(text.split())
    
    # Look for author pattern after title - typically names followed by affiliations
    # Pattern: Title followed by authors (names with possible middle initials)
    author_pattern = r'(?:Short communication|Research report|Brief report|Review article)\s+[^a-z]*?\s+([A-Z][a-z]+\s+[A-Z]?[a-z]*,?\s*(?:[A-Z][a-z]+\s+[A-Z]?[a-z]*,?\s*)*)\s*[a-z,]*\s*(?:Department|Centre|Institute|University)'
    
    match = re.search(author_pattern, clean_text, re.IGNORECASE)
    if match:
        authors = match.group(1).strip()
        # Clean up authors
        authors = re.sub(r'[0-9*†‡§¶#]', '', authors)
        authors = re.sub(r'\s+', ' ', authors).strip()
        if len(authors) > 5:
            return authors
    
    return None

def extract_journal_from_text(text):
    """
    Extract journal name - simplified
    """
    if 'behavioural brain research' in text.lower():
        return 'Behavioural Brain Research'
    
    # Look for other journal patterns
    journal_patterns = [
        r'(Journal of [A-Z][a-z\s]+)',
        r'([A-Z][a-z]+\s+(?:Research|Science|Psychology|Neuroscience))',
    ]
    
    for pattern in journal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def extract_metadata_from_pdf(pdf_path):
    """
    Extract comprehensive metadata including title, authors, journal, and year from PDF
    """
    metadata = {
        'source': pdf_path.name,
        'year': None,
        'title': None,
        'authors': None,
        'journal': None
    }
    
    try:
        doc = fitz.open(str(pdf_path))
        
        # Try to get title from PDF metadata first
        pdf_metadata = doc.metadata
        if pdf_metadata and pdf_metadata.get('title'):
            title = pdf_metadata['title'].strip()
            if len(title) > 5:
                metadata['title'] = title[:300]
        
        if len(doc) > 0:
            first_page = doc[0].get_text()
            
            # Extract year
            year = extract_year_from_text(first_page)
            if year:
                metadata['year'] = year
            
            # Extract title if not found in metadata
            if not metadata['title']:
                title = extract_title_from_text(first_page)
                if title:
                    metadata['title'] = title
            
            # Extract authors
            authors = extract_authors_from_text(first_page)
            if authors:
                metadata['authors'] = authors
            
            # Extract journal
            journal = extract_journal_from_text(first_page)
            if journal:
                metadata['journal'] = journal
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting metadata from {pdf_path.name}: {e}")
    
    return metadata

def load_documents_with_metadata(input_dir="./data/apa-papers"):
    """
    Load documents from PDF files and extract metadata
    """
    pdf_files = list(Path(input_dir).glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    documents = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(pdf_path)
            print(f"  Title: {metadata['title'][:50] + '...' if metadata['title'] and len(metadata['title']) > 50 else metadata['title']}")
            print(f"  Year: {metadata['year']}")
            print(f"  Journal: {metadata['journal']}")
            
            # Load document content using SimpleDirectoryReader
            loader = SimpleDirectoryReader(input_files=[str(pdf_path)])
            docs = loader.load_data()
            
            # Add metadata to each document chunk
            for doc in docs:
                doc.metadata.update(metadata)
                documents.append(doc)
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    return documents

def save_unique_metadata_summary(documents, output_file="psychology_metadata.jsonl"):
    """
    Save extracted metadata to JSONL file for inspection - one entry per unique PDF
    """
    print(f"\nSaving metadata summary to {output_file}...")
    
    # Group by source (PDF filename) to get unique papers
    unique_papers = {}
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        if source not in unique_papers:
            unique_papers[source] = {
                'source': source,
                'year': doc.metadata.get('year', None),
                'title': doc.metadata.get('title', None),
                'authors': doc.metadata.get('authors', None),
                'journal': doc.metadata.get('journal', None)
            }
    
    # Convert to list for consistent processing
    metadata_list = list(unique_papers.values())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Statistics
    years_found = [entry['year'] for entry in metadata_list if entry['year']]
    titles_found = [entry['title'] for entry in metadata_list if entry['title']]
    authors_found = [entry['authors'] for entry in metadata_list if entry['authors']]
    journals_found = [entry['journal'] for entry in metadata_list if entry['journal']]
    
    print(f"✅ Metadata saved for {len(metadata_list)} unique papers")
    print(f"Titles extracted: {len(titles_found)}/{len(metadata_list)} papers")
    print(f"Authors extracted: {len(authors_found)}/{len(metadata_list)} papers")
    print(f"Journals extracted: {len(journals_found)}/{len(metadata_list)} papers")

# Main execution
if __name__ == "__main__":
    print("Building vector database with metadata extraction...")
    
    # Step 1: Load documents with metadata
    documents = load_documents_with_metadata("./data/apa-papers")
    print(f"\nTotal document chunks loaded: {len(documents)}")
    
    # Step 2: Set up embedding model and chunking settings
    embed_model = HuggingFaceEmbedding()
    Settings.chunk_size = 512 
    Settings.chunk_overlap = 50
    
    # Step 3: Set up persistent ChromaDB vector store
    db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Step 4: Build and persist index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    print("\n✅ Vector database created with metadata!")
    
    # Step 5: Save metadata summary
    save_unique_metadata_summary(documents)
    
    print("\n🎉 Complete! Your RAG system now has rich metadata for all papers.")
