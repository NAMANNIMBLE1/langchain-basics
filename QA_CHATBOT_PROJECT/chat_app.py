import os 
from langchain_groq import  ChatGroq
from dotenv import load_dotenv
import streamlit as st
# import openai  # if we need to work with open ai chat open ai 
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")



system_prompt = (
    "you are a assistant for question answer tasks"
    "use the following recieved context data to answer"
    "if you dont know the answer say that you don't know"
    "keep the answer consise straight to the point"
)

## prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","Question:{question}")
    ]
)


def generate_response(question , groq_api_key , llms , temprature , max_tokens):
    groq_api_key = groq_api_key
    llm = ChatGroq(model=llms, groq_api_key=groq_api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {'question' : question}
    )
    return answer 


#################################### streamlit ############################################
st.title("Q&A BOT")
# different models 
llms = st.sidebar.selectbox("Select AI Model" ,["llama-guard-3-8b",
                                                "qwen-2.5-32b",
                                                "deepseek-r1-distill-llama-70b",
                                                "llama-3.1-8b-instant",
                                                "mistral-saba-24b",
                                                "llama-3.2-1b-preview",
                                                "llama3-70b-8192",
                                                "qwen-2.5-coder-32b",
                                                "llama-3.3-70b-specdec",
                                                "gemma2-9b-it",
                                                "deepseek-r1-distill-qwen-32b",
                                                "allam-2-7b",
                                                "qwen-qwq-32b",
                                                "llama3-8b-8192",
                                                "llama-3.2-11b-vision-preview",
                                                "llama-3.3-70b-versatile",
                                                "distil-whisper-large-v3-en",
                                                "llama-3.2-3b-preview",
                                                "llama-3.2-90b-vision-preview",
                                                "whisper-large-v3-turbo"
                                            ]
                                )

# a=djust response parameter 
st.sidebar.title("Settings")
grok_api_key = st.sidebar.text_input("enter your Groq ai api key" , type="password")
temprature = st.sidebar.slider("Temprature" , min_value = 0.0 , max_value=1.0 , value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens" , min_value = 50 , max_value=300 , value = 150)


# main interface input 
st.write("Ask anything what comes to your mind :)")
user_input = st.text_input("You:")
button = st.button("submit")

if user_input and button:
    response = generate_response(user_input , grok_api_key , llms, temprature , max_tokens)
    st.write(response)
else:
    st.write("please provide the query")