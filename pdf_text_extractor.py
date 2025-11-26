#!/usr/bin/env python3
"""
PDF Text Extractor

A simple script to extract text from PDF files using PyPDF2.
Supports both single file and batch processing.
"""

import argparse
import os
from pathlib import Path
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Install with: pip install PyPDF2")
    exit(1)


def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def save_text_to_file(text, output_path):
    """Save extracted text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to: {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("input", help="PDF file or directory containing PDF files")
    parser.add_argument("-o", "--output", help="Output file or directory (optional)")
    parser.add_argument("--print", action="store_true", help="Print text to console")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Single PDF file
        print(f"Extracting text from: {input_path}")
        text = extract_text_from_pdf(input_path)
        
        if text:
            if args.print:
                print("\n" + "="*50)
                print(text)
            
            if args.output:
                output_path = Path(args.output)
                if output_path.suffix == '':
                    output_path = output_path / f"{input_path.stem}.txt"
            else:
                output_path = input_path.with_suffix('.txt')
            
            save_text_to_file(text, output_path)
    
    elif input_path.is_dir():
        # Directory with PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_path}")
            return
        
        output_dir = Path(args.output) if args.output else input_path
        output_dir.mkdir(exist_ok=True)
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file}")
            text = extract_text_from_pdf(pdf_file)
            
            if text:
                if args.print:
                    print(f"\n{'='*20} {pdf_file.name} {'='*20}")
                    print(text[:500] + "..." if len(text) > 500 else text)
                
                output_file = output_dir / f"{pdf_file.stem}.txt"
                save_text_to_file(text, output_file)
    
    else:
        print(f"Invalid input: {input_path}")
        print("Please provide a PDF file or directory containing PDF files")


if __name__ == "__main__":
    main()