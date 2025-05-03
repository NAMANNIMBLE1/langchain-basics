from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chat with Ollama"

# prompt template 
system_prompt = (
    "you are a assistant for question answer tasks"
    "use the following recieved context data to answer"
    "if you dont know the answer say that you don't know"
    "keep the answer consise straight to the point"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","Question:{question}")
    ]
)

def generate_response(question , engine):
    llm = Ollama(
        model = engine
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {'question' : question}
    )
    return answer 

#################################### streamlit ############################################
st.title("Q&A BOT")


# main interface input 
st.write("Ask anything what comes to your mind :)")
user_input = st.text_input("You:")
button = st.button("submit")

engine = st.sidebar.selectbox("Select Engine", ['gemma2', 'llama3'])

if user_input and button:
    response = generate_response(user_input , engine)
    st.write(response)
else:
    st.write("please provide the query")