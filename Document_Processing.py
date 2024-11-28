# document_processing_app.py

import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_directory(directory_path):
    """Load markdown documents from a directory and return a list of Document objects."""
    documents = []
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()  # Remove leading/trailing whitespace
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": filename}))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not documents:
        raise ValueError(f"No valid documents found in '{directory_path}'.")
    
    print(f"Loaded {len(documents)} documents from '{directory_path}'.")
    return documents

def split_documents(documents, chunk_size=1500, chunk_overlap=50):
    """Split documents into chunks of a specified size with overlapping text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    
    if not texts:
        raise ValueError("No chunks were generated from the documents.")
    
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")
    return texts

def save_chunks_to_file(chunks, output_directory):
    """Save text chunks to separate files in the specified directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_directory, f"chunk_{i + 1}.txt")  # Naming each chunk file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(chunk.page_content)
            print(f"Saved chunk {i + 1} to '{output_file}'.")
        except Exception as e:
            print(f"Error saving chunk {i + 1} to {output_file}: {e}")

if __name__ == "__main__":
    # Configuration
    markdown_directory = "output"
    output_directory = "text_chunks"
    chunk_size = 1500
    chunk_overlap = 200

    # Load documents
    try:
        documents = load_documents_from_directory(markdown_directory)
    except Exception as e:
        print(f"Error loading documents: {e}")
        exit(1)

    # Optional: Inspect a sample of loaded documents (debugging)
    for i, doc in enumerate(documents[:5]):  # Limit to first 5 documents
        print(f"Document {i} from '{doc.metadata['source']}': {doc.page_content[:100]}...\n")

    # Split documents into chunks
    try:
        chunks = split_documents(documents, chunk_size, chunk_overlap)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        exit(1)

    # Save text chunks to separate files
    save_chunks_to_file(chunks, output_directory)
