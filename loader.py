import os
from langchain.document_loaders import PyPDFLoader, TextLoader

def validate_pdf_file(pdf_path):
    """Validate that the file exists and is a PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"File is not a PDF: {pdf_path}")

    # Check if file is readable
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                raise ValueError(f"File does not appear to be a valid PDF: {pdf_path}")
    except Exception as e:
        raise ValueError(f"Cannot read PDF file {pdf_path}: {e}")

    print(f"PDF file validated: {pdf_path}")

def load_document(file_path):
    print(f"Loading document: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()

    documents = []
    try:
        if file_extension == ".pdf":
            print("Attempting PyPDFLoader...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if documents:
                print(f"PyPDFLoader successful - {len(documents)} pages loaded")
        elif file_extension == ".txt":
            print("Attempting TextLoader...")
            loader = TextLoader(file_path)
            documents = loader.load()
            if documents:
                print(f"TextLoader successful - {len(documents)} documents loaded")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported.")

        if not documents:
            raise RuntimeError("Document loading failed: No content loaded.")

        return documents

    except Exception as e:
        print(f"Document loading failed for {file_path}: {e}")
        raise

def preprocess_document(documents):
    print("Preprocessing documents...")

    processed_docs = []
    for doc in documents:
        text = doc.page_content

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove page headers/footers if they're repetitive
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip very short lines that might be artifacts
            if len(line) > 3:
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # Only keep documents with substantial content
        if len(cleaned_text.strip()) > 50:
            doc.page_content = cleaned_text
            processed_docs.append(doc)

    print(f"Processed {len(processed_docs)} documents after cleaning")
    return processed_docs