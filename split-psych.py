import json
import os
import re
from pathlib import Path
from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
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

def extract_metadata_from_pdf(pdf_path):
    """
    Extract metadata including year from the first page of PDF
    """
    metadata = {
        'source': pdf_path.name,
        'year': None,
        'title': None
    }
    
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) > 0:
            first_page = doc[0].get_text()
            
            year = extract_year_from_text(first_page)
            if year:
                metadata['year'] = year
            
            lines = first_page.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.lower().startswith(('abstract', 'keywords', 'introduction')):
                    metadata['title'] = line[:200]  # Limit title length
                    break
        
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
            metadata = extract_metadata_from_pdf(pdf_path)
            print(f"  Year extracted: {metadata['year']}")
            
            loader = SimpleDirectoryReader(input_files=[str(pdf_path)])
            docs = loader.load_data()
            
            for doc in docs:
                doc.metadata.update(metadata)
                documents.append(doc)
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    return documents

def build_vectordb_with_metadata():
    """
    Build vector database similar to build-vectordb.py but with metadata extraction
    """
    print("Starting vector database creation with metadata...")
    
    documents = load_documents_with_metadata("./data/apa-papers")
    print(f"Total documents loaded: {len(documents)}")
    
    print("\nMetadata examples:")
    for i, doc in enumerate(documents[:5]):
        print(f"  Doc {i+1}: Year={doc.metadata.get('year', 'N/A')}, Source={doc.metadata.get('source', 'N/A')}")
    
    embed_model = HuggingFaceEmbedding()
    
    Settings.chunk_size = 512 
    Settings.chunk_overlap = 50
    
    db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    print("Vector database created with metadata!")
    
    save_metadata_summary(documents)
    
    return index

def save_metadata_summary(documents, output_file="psychology_metadata.jsonl"):
    """
    Save extracted metadata to JSONL file for inspection
    """
    print(f"\nSaving metadata to {output_file}...")
    
    metadata_list = []
    for doc in documents:
        metadata_entry = {
            'source': doc.metadata.get('source', 'Unknown'),
            'year': doc.metadata.get('year', None),
            'title': doc.metadata.get('title', None),
            'text_preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        }
        metadata_list.append(metadata_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    years_found = [entry['year'] for entry in metadata_list if entry['year']]
    print(f"✅ Metadata saved for {len(metadata_list)} documents")
    print(f"Years extracted from {len(years_found)} documents")
    if years_found:
        unique_years = sorted(set(years_found))
        print(f"Year range: {min(unique_years)} - {max(unique_years)}")
        print(f"Unique years: {unique_years}")

if __name__ == "__main__":
    print("Testing metadata extraction...")
    pdf_files = list(Path("./data/apa-papers").glob("**/*.pdf"))
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"Testing with: {test_pdf.name}")
        metadata = extract_metadata_from_pdf(test_pdf)
        print(f"Extracted metadata: {metadata}")
        print()
    
    index = build_vectordb_with_metadata()
