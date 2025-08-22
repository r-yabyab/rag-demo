import fitz
from pathlib import Path
import sys
sys.path.append('.')
from split_psych import extract_metadata_from_pdf

# Test with one specific PDF
test_file = Path('./data/apa-papers/behavioural brain research volume 272/Social-status-and-GnRH-soma-size-in-female-convict-cich_2014_Behavioural-Bra.pdf')

if test_file.exists():
    print(f'Testing with: {test_file.name}')
    
    # Extract metadata using our function
    metadata = extract_metadata_from_pdf(test_file)
    print('\n=== Extracted metadata ===')
    for key, value in metadata.items():
        print(f'{key}: {value}')
    
    print('\n=== Testing another PDF ===')
    
    # Test with another file
    test_file2 = Path('./data/apa-papers/behavioural brain research volume 272/3D-video-analysis-of-the-novel-object-recognition-_2014_Behavioural-Brain-Re.pdf')
    if test_file2.exists():
        metadata2 = extract_metadata_from_pdf(test_file2)
        print(f'Testing with: {test_file2.name}')
        for key, value in metadata2.items():
            print(f'{key}: {value}')
    
else:
    print('File not found')
