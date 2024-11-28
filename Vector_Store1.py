# vector_store_app.py

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_text_chunks(directory="text_chunks"):
    """Load text chunks from a directory."""
    text_chunks = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # Ignore empty files or content
                        text_chunks.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not text_chunks:
        raise ValueError(f"No valid text chunks found in '{directory}'.")
    
    print(f"Loaded {len(text_chunks)} text chunks from '{directory}'.")
    return text_chunks

def create_vector_store(texts, embedding_model_name="sentence-transformers/msmarco-distilbert-base-v4"):
    """Create a FAISS vector store from a list of text chunks."""
    if not texts:
        raise ValueError("No texts provided to create vector store.")
    
    try:
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Create vector store
        vector_store = FAISS.from_texts(texts, embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error during vector store creation: {e}")

def save_vector_store(vector_store, directory="faiss_index", index_name="faiss_index"):
    """Save the FAISS vector store to disk."""
    os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, index_name)
    try:
        vector_store.save_local(index_path)
        print(f"Vector store saved to '{index_path}'.")
    except Exception as e:
        raise RuntimeError(f"Error saving vector store to disk: {e}")

if __name__ == "__main__":
    try:
        # Load text chunks from the text_chunks directory
        sample_texts = load_text_chunks()

        # Create vector store
        vector_store = create_vector_store(sample_texts)

        # Save vector store to disk
        save_vector_store(vector_store)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
