import streamlit as st 
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings , ChatNVIDIA 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

## load nvidia api key  

# os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(
    model = "meta/llama-3.2-3b-instruct", # not giving api key as we are already loading it
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()
        
        # Correct usage of RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
        st.session_state.final_documents = splitter.split_documents(st.session_state.docs)
        
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, 
            st.session_state.embeddings
        )



st.title("NVIDIA DEMO NIM ")

# Define the actual prompt template
chat_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Now get the user's question
user_question = st.text_input("Enter your question based on the documents")


if st.button("document embedding"):
    vector_embeddings()
    st.write("vector store db is ready using nvidia embeddings")


if user_question:
    document_chain = create_stuff_documents_chain(llm, chat_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({
        "input": user_question
    })
    print("response time:", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------------")
