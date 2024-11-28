# vector_store_app.py

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(texts, embedding_model_name="sentence-transformers/msmarco-distilbert-base-v4"):
    if not texts:
        raise ValueError("No texts found to create vector store")
    
    try:
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # FAISS expects raw text, so we pass the list of strings directly
        vector_store = FAISS.from_texts(texts, embeddings)
        print("Vector store created successfully")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error during vector store creation: {e}")

if __name__ == "__main__":
    # Load text chunks from the file created in Document_Processing.py
    try:
        with open("text_chunks.txt", "r", encoding="utf-8") as f:
            sample_texts = f.read().splitlines()
        
        # Create vector store
        vector_store = create_vector_store(sample_texts)
    except FileNotFoundError:
        print("Text chunks file not found. Please ensure Document_Processing.py has run successfully.")
    
