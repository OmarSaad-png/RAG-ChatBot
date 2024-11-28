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

# Environment variables for Hugging Face API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# Function to load text chunks from files
def load_text_chunks(directory="text_chunks"):
    text_chunks = []
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist.")
        st.error(f"Directory '{directory}' does not exist.")
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    text_chunks.append(content)
                else:
                    logger.warning(f"File {file_path} is empty and will be skipped.")
    return text_chunks


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

    # Define prompts
    condense_question_template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone question:
    """
    condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

    qa_template = """
    You are an researcher in Reinforcement learning required to answer questions.
    Use the following pieces of retrieved context to answer the question and don't halucinate.
    If you don't know the answer, say that you don't know.

    Chat History:
    {chat_history}
    Other context:
    {context}
    Question: {question}
    """
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        memory=memory,
    )

    return conversation_chain

texts = load_text_chunks()

if not texts:
    st.error("No text chunks available for creating the vector store.")
else:
    vector_store = create_vector_store(texts)  # Create vector store
    conversation_chain = get_conversational_chain(vector_store)  # Initialize conversation_chain here

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Fetching response..."):
        response = conversation_chain.run(query)
        if response:
            st.markdown(f"**Answer:** {response}")
        else:
            st.error("No response generated.")

# Streamlit App
st.title("Reinforcement learning ChatBot")
