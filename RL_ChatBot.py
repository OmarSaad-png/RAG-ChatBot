import streamlit as st
import os
import logging
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from Vector_Store1 import create_vector_store  
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit caching for efficiency
@st.cache_resource
def create_cached_vector_store(texts):
    return create_vector_store(texts)

def load_text_chunks_parallel(directory="text_chunks"):
    text_chunks = []
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist.")
        st.error(f"Directory '{directory}' does not exist.")
        return []

    with ThreadPoolExecutor() as executor:
        text_chunks = list(executor.map(read_file, os.listdir(directory)))
    
    text_chunks = [chunk for sublist in text_chunks for chunk in sublist if chunk]
    return text_chunks

def read_file(filename):
    file_chunks = []
    if filename.endswith(".txt"):
        file_path = os.path.join("text_chunks", filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                file_chunks.append(content)
            else:
                logger.warning(f"File {file_path} is empty and will be skipped.")
    return file_chunks

# Initialize LLM and conversational chain
def get_conversational_chain(vector_store):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=256,
        do_sample=True,
        repetition_penalty=1.03,
        temperature=0.7,
    )
    chat_model = ChatHuggingFace(llm=llm)

    condense_question_template = """
    Given the following conversation on Reinforcement Learning and a follow-up question, prompt engineer the follow-up question to be a standalone question.
    Chat History: {chat_history}
    Follow-Up Input: {question}
    Standalone question:
    """
    condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

    qa_template = """
    You are a researcher in Reinforcement learning. Use the following pieces of retrieved context to answer the question, make tour answer concise. If you don't know the answer, say that you don't know, don't halucinate.
    Chat History: {chat_history}
    Other context: {context}
    Question: {question}
    """
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        memory=memory,
    )

    return conversation_chain

# Main app logic
texts = load_text_chunks_parallel()

if not texts:
    st.error("No text chunks available for creating the vector store.")
else:
    vector_store = create_cached_vector_store(texts)  # Cached vector store
    conversation_chain = get_conversational_chain(vector_store)

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
