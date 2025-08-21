import json
import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_psychology_journals(input_dir="data/apa-papers", output_file="psychology_sections.jsonl", max_tokens=500):
    """
    Split psychology journal PDFs into sections under max_tokens and save to JSONL
    
    Args:
        input_dir: Directory containing PDF files
        output_file: Output JSONL file path
        max_tokens: Maximum tokens per section
    """
    
    # Initialize text splitter with token-based splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # Rough estimate: 1 token ≈ 4 characters
        chunk_overlap=50,
        length_function=count_tokens,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
    )
    
    pdf_files = list(Path(input_dir).glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    subdirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    if subdirs:
        print(f"Searching in subdirectories: {[d.name for d in subdirs]}")
    
    sections = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            full_text = "\n".join([doc.page_content for doc in documents])
            
            chunks = text_splitter.split_text(full_text)
            
            for chunk in chunks:
                token_count = count_tokens(chunk)
                
                if token_count <= max_tokens:
                    cleaned_text = chunk.strip()
                    cleaned_text = cleaned_text.lstrip('.,;:!? \n\t') 
                    
                    if cleaned_text and len(cleaned_text) > 10:
                        sections.append({
                            "text": cleaned_text,
                            "source": pdf_path.name,
                            "token_count": count_tokens(cleaned_text)
                        })
                else:
                    smaller_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=max_tokens * 3,
                        chunk_overlap=20,
                        length_function=count_tokens,
                        separators=[". ", "! ", "? ", "; ", " "]
                    )
                    smaller_chunks = smaller_splitter.split_text(chunk)
                    
                    for small_chunk in smaller_chunks:
                        small_token_count = count_tokens(small_chunk)
                        if small_token_count <= max_tokens:
                            cleaned_text = small_chunk.strip()
                            cleaned_text = cleaned_text.lstrip('.,;:!? \n\t') 
                            
                            if cleaned_text and len(cleaned_text) > 10:
                                sections.append({
                                    "text": cleaned_text,
                                    "source": pdf_path.name,
                                    "token_count": count_tokens(cleaned_text)
                                })
            
            print(f"  Extracted {len([s for s in sections if s['source'] == pdf_path.name])} sections")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    print(f"\nSaving {len(sections)} sections to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for section in sections:
            json_line = {"text": section["text"]}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully saved {len(sections)} sections to {output_file}")
    
    token_counts = [section["token_count"] for section in sections]
    if token_counts:
        print(f"\nToken Statistics:")
        print(f"  Average tokens per section: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Sections under {max_tokens} tokens: {len([t for t in token_counts if t <= max_tokens])}/{len(token_counts)}")

if __name__ == "__main__":
    INPUT_DIR = "data/apa-papers"
    OUTPUT_FILE = "psychology_sections.jsonl" 
    MAX_TOKENS = 500
    
    split_psychology_journals(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        max_tokens=MAX_TOKENS
    )