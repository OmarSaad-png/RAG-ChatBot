# document_processing_app.py

import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # Ensure content is not empty
                    documents.append(Document(page_content=content))
    print(f"Loaded {len(documents)} documents from {directory_path}")  # Debugging
    return documents

def validate_document_content(doc):
    return bool(doc.page_content.strip())

def split_documents(documents, chunk_size=1500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")  # Debugging
    return texts

if __name__ == "__main__":
    markdown_directory = "output"
    documents = load_documents_from_directory(markdown_directory)
    if not documents:
        raise ValueError("No documents loaded from the directory")

    # Inspect loaded documents
    for i, doc in enumerate(documents):
        print(f"Document {i}: {doc.page_content[:100]}...")  # Debugging

    texts = split_documents(documents)
    if not texts:
        raise ValueError("No chunks generated from the documents")

    # Save texts in vector store creation
    with open("text_chunks.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.page_content + "\n")
