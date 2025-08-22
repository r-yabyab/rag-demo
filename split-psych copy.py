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

def extract_title_from_text(text):
    """
    Extract paper title from the first page text
    """
    lines = text.split('\n')
    
    # Skip headers, footers, and common metadata lines
    skip_patterns = [
        r'^(contents|available online|published by|elsevier|doi:|http|www\.)',
        r'^\d+$',  # page numbers
        r'^[A-Z\s]{2,}$',  # ALL CAPS headers
        r'(journal|volume|issue|page|\d{4}.*elsevier)',
        r'^(research paper|review|article|correspondence)',
        r'^(received|accepted|available|published)',
    ]
    
    potential_titles = []
    
    for i, line in enumerate(lines[:20]):  # Check first 20 lines
        line = line.strip()
        
        # Skip empty lines and very short lines
        if len(line) < 10:
            continue
            
        # Skip lines matching skip patterns
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
        
        # Look for title characteristics
        if (len(line) > 20 and 
            not line.isupper() and 
            not line.lower().startswith(('abstract', 'keywords', 'introduction', 'background'))):
            potential_titles.append((i, line))
    
    # Return the first suitable title found
    if potential_titles:
        return potential_titles[0][1][:300]  # Limit title length
    
    return None

def extract_authors_from_text(text):
    """
    Extract authors from the first page text
    """
    lines = text.split('\n')
    
    # Common patterns for author sections
    author_indicators = [
        r'^[A-Z][a-z]+\s+[A-Z]\.',  # LastName F.
        r'^[A-Z][a-z]+,\s*[A-Z]\.',  # LastName, F.
        r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[a-z]*',  # Full names
    ]
    
    authors = []
    in_author_section = False
    
    for i, line in enumerate(lines[:30]):  # Check first 30 lines
        line = line.strip()
        
        # Skip very short lines
        if len(line) < 3:
            continue
            
        # Look for author patterns
        for pattern in author_indicators:
            if re.search(pattern, line):
                # Clean up the author line
                author_line = re.sub(r'[0-9*†‡§¶#]', '', line)  # Remove superscript numbers/symbols
                author_line = re.sub(r'\s+', ' ', author_line).strip()
                
                if len(author_line) > 2 and len(author_line) < 200:
                    authors.append(author_line)
                    in_author_section = True
                break
        
        # If we found authors and hit a line that doesn't look like authors, stop
        if in_author_section and not any(re.search(pattern, line) for pattern in author_indicators):
            if not re.search(r'[a-z]', line):  # If line has no lowercase (likely not authors)
                break
    
    # Join authors if multiple lines found
    if authors:
        return '; '.join(authors[:5])  # Limit to first 5 author entries
    
    return None

def extract_journal_from_text(text):
    """
    Extract journal name from the text
    """
    lines = text.split('\n')
    
    # Common journal patterns
    journal_patterns = [
        r'(Behavioural Brain Research|Brain Research|Neuroscience|Psychology|Journal of)',
        r'^\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\d+',  # Journal Name + volume
        r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:Volume|Vol\.)',  # Journal + Volume
    ]
    
    for line in lines[:15]:  # Check first 15 lines
        line = line.strip()
        
        for pattern in journal_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                journal = match.group(1) if match.lastindex else match.group(0)
                # Clean up the journal name
                journal = re.sub(r'\d+.*$', '', journal).strip()  # Remove volume/page numbers
                if len(journal) > 5 and len(journal) < 100:
                    return journal
    
    # Fallback: look for "Behavioural Brain Research" specifically since that's in your folder
    if 'behavioural brain research' in text.lower():
        return 'Behavioural Brain Research'
    
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
        if len(doc) > 0:
            first_page = doc[0].get_text()
            
            # Extract year
            year = extract_year_from_text(first_page)
            if year:
                metadata['year'] = year
            
            # Extract title
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
            metadata = extract_metadata_from_pdf(pdf_path)
            print(f"  Year: {metadata['year']}")
            print(f"  Title: {metadata['title'][:50] + '...' if metadata['title'] and len(metadata['title']) > 50 else metadata['title']}")
            print(f"  Authors: {metadata['authors'][:50] + '...' if metadata['authors'] and len(metadata['authors']) > 50 else metadata['authors']}")
            print(f"  Journal: {metadata['journal']}")
            
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
    for i, doc in enumerate(documents[:3]):
        print(f"  Doc {i+1}:")
        print(f"    Year: {doc.metadata.get('year', 'N/A')}")
        print(f"    Title: {doc.metadata.get('title', 'N/A')[:60]}{'...' if doc.metadata.get('title') and len(doc.metadata.get('title', '')) > 60 else ''}")
        print(f"    Authors: {doc.metadata.get('authors', 'N/A')[:50]}{'...' if doc.metadata.get('authors') and len(doc.metadata.get('authors', '')) > 50 else ''}")
        print(f"    Journal: {doc.metadata.get('journal', 'N/A')}")
        print(f"    Source: {doc.metadata.get('source', 'N/A')}")
        print()
    
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
            'authors': doc.metadata.get('authors', None),
            'journal': doc.metadata.get('journal', None),
            'text_preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        }
        metadata_list.append(metadata_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    years_found = [entry['year'] for entry in metadata_list if entry['year']]
    titles_found = [entry['title'] for entry in metadata_list if entry['title']]
    authors_found = [entry['authors'] for entry in metadata_list if entry['authors']]
    journals_found = [entry['journal'] for entry in metadata_list if entry['journal']]
    
    print(f"✅ Metadata saved for {len(metadata_list)} documents")
    print(f"Years extracted from {len(years_found)} documents")
    print(f"Titles extracted from {len(titles_found)} documents")
    print(f"Authors extracted from {len(authors_found)} documents")
    print(f"Journals extracted from {len(journals_found)} documents")
    
    if years_found:
        unique_years = sorted(set(years_found))
        print(f"Year range: {min(unique_years)} - {max(unique_years)}")
        print(f"Unique years: {unique_years}")
    
    if journals_found:
        unique_journals = set(journals_found)
        print(f"Unique journals found: {list(unique_journals)}")

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
