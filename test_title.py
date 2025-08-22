import fitz
from pathlib import Path

def simple_extract_title(pdf_path):
    """
    Simple title extraction focusing on PDF metadata and basic text patterns
    """
    try:
        doc = fitz.open(str(pdf_path))
        
        # Strategy 1: Try PDF metadata first
        metadata = doc.metadata
        if metadata and metadata.get('title'):
            title = metadata['title'].strip()
            if len(title) > 5:
                doc.close()
                return title
        
        # Strategy 2: Get text and join it properly
        if len(doc) > 0:
            text = doc[0].get_text()
            # Join all words with spaces
            clean_text = ' '.join(text.split())
            
            # Look for patterns after "Short communication" or "Research report"
            import re
            patterns = [
                r'Short communication\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
                r'Research report\s+(.+?)(?:\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+[a-z])|Abstract|Department)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, clean_text, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Clean up the title
                    title = re.sub(r'\s+', ' ', title)
                    if 10 < len(title) < 200:
                        doc.close()
                        return title
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting title from {pdf_path.name}: {e}")
    
    return None

# Test with the problem file
test_file = Path('./data/apa-papers/behavioural brain research volume 272/Social-status-and-GnRH-soma-size-in-female-convict-cich_2014_Behavioural-Bra.pdf')

if test_file.exists():
    print(f'Testing title extraction with: {test_file.name}')
    title = simple_extract_title(test_file)
    print(f'Extracted title: {title}')
    
    # Let's also show the cleaned text to debug
    doc = fitz.open(str(test_file))
    text = doc[0].get_text()
    clean_text = ' '.join(text.split())
    print(f'\nFirst 500 chars of cleaned text:')
    print(clean_text[:500])
    doc.close()
else:
    print('Test file not found')
