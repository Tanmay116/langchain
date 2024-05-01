from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import load_tools ,initialize_agent

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title=f"Streaming Bot")

st.title("Streaming Bot")

# get response
def get_response(query, chat_history):
    template = """
    You are a helpful assistant. Please response to the user queries
    Chat history: {chat_history}
    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(temperature=0, model_name="gemma-7b-it")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })

# coversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


user_query = st.chat_input("Your Message")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))