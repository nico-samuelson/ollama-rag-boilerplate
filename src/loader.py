import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def validate_pdf_file(self) -> None:
        """Validate that the file exists and is a PDF."""

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if not self.file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {self.file_path}")

        # Check if file is readable
        try:
            with open(self.file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    raise ValueError(f"File does not appear to be a valid PDF: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Cannot read PDF file {self.file_path}: {e}")

        print(f"PDF file validated: {self.file_path}")

    def load_document(self) -> List[Document]:
        """
        Load the document based on its file type. Supports PDF and TXT files.

        Returns:
            list: List of Document objects loaded from the file.
        """

        print(f"Loading document: {self.file_path}")
        file_extension = os.path.splitext(self.file_path)[1].lower()

        documents = []
        try:
            if file_extension == ".pdf":
                # Validate PDF before loading
                self.validate_pdf_file()

                print("Attempting PyPDFLoader...")
                loader = PyPDFLoader(self.file_path)
                documents = loader.load()
                if documents:
                    print(f"PyPDFLoader successful - {len(documents)} pages loaded")
            elif file_extension == ".txt":
                print("Attempting TextLoader...")
                loader = TextLoader(self.file_path)
                documents = loader.load()
                if documents:
                    print(f"TextLoader successful - {len(documents)} documents loaded")
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt are supported.")

            if not documents:
                raise RuntimeError("Document loading failed: No content loaded.")

            return documents

        except Exception as e:
            print(f"Document loading failed for {self.file_path}: {e}")
            raise

    def preprocess_document(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents to clean and filter content.

        Args:
            documents (list): List of Document objects to preprocess
        Returns:
            list: List of cleaned Document objects
        """

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