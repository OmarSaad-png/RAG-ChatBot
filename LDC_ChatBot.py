import streamlit as st
import requests
import logging
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import ConversationalRetrievalChain
from Vector_Store1 import create_vector_store
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

API_BASE_URL = "http://localhost:5000/api/products"

# Function to load text chunks
def load_text_chunks(file_path="text_chunks.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = f.read().splitlines()
        return texts
    except FileNotFoundError:
        logger.error("Text chunks file not found.")
        st.error("Text chunks file not found.")
        return []

# API Interaction Functions
def fetch_all_products():
    try:
        response = requests.get(f'{API_BASE_URL}/getAll')
        response.raise_for_status()
        products = response.json()
        logger.info("Fetched all products successfully.")
        return products
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching all products: {e}")
        st.error(f"Error fetching all products: {e}")
        return None

def fetch_product_by_name(product_name):
    url = f'{API_BASE_URL}/getName/{product_name}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        product = response.json()
        if product:
            logger.info(f"Product '{product_name}' fetched successfully.")
            return product
        else:
            logger.warning(f"Product '{product_name}' not found.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching product '{product_name}': {e}")
        st.error(f"Error fetching product '{product_name}': {e}")
        return None

# Query Parsing and Processing
def format_product_response(product):
    product_info = (
        f"**Product Name:** {product['name']}\n"
        f"**Description:** {product['description']}\n"
        f"**Price:** ${product['price']}\n"
        f"**Stock:** {product['stock']} units"
    )
    logger.info("Product information formatted successfully.")
    return product_info

# Initialize LLM and conversational chain
def get_conversational_chain(vector_store):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
        repetition_penalty=1.03,
        temperature=0.7,
    )

    chat_model = ChatHuggingFace(llm=llm)

    condense_question_template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone question:"""

    condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

    qa_template = """
    You are an assistant for LDC's company required to answer questions.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know.

    Chat History:
    {chat_history}

    Other context:
    {context}

    Question: {question}
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        memory=memory,
    )

    return conversation_chain

# Load text chunks once at the start
texts = load_text_chunks()
if not texts:
    st.error("No text chunks available for creating the vector store.")
else:
    # Create vector store once and reuse it
    vector_store = create_vector_store(texts)  # Store it in memory
    conversation_chain = get_conversational_chain(vector_store)

    # Streamlit interface
    st.title("LDC ChatBot")

    # Query input
    query = st.text_input("Enter your question:")

    if query:
        response = conversation_chain.run(query)
        st.markdown(f"**Answer:** {response}")
 
   