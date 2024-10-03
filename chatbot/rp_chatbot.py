import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.schema import HumanMessage, AIMessage  # Import the relevant message classes
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
LLM = "llama3-groq-70b-8192-tool-use-preview"
parser = StrOutputParser()
# Set up the chatbot and tools
tvly = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=50000, load_all_available_meta=True, load_max_docs=10)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

memory = MemorySaver()
tools = [arxiv_tool, wikipedia_tool, tvly]

llm = ChatGroq(temperature=0, model_name="llama3-groq-70b-8192-tool-use-preview")
llm_with_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
tool_node = ToolNode(tools=tools)

# Define chatbot logic
def chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

# Streamlit UI
st.title("LangGraph Research Paper Chatbot:smile:")

# Initialize the session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("You: ")

with st.sidebar:
    LLM = st.selectbox(
            "Choose the llm of your preference!",
            ("gemma2-9b-it", "llama3-groq-70b-8192-tool-use-preview",
              "mixtral-8x7b-32768", "llama-3.1-70b-versatile")
        )

# Chatbot interaction
if user_input is not None and user_input != "":
    st.session_state["messages"].append(("user", user_input))
    config = {"configurable": {"thread_id": "126"}}
    events = graph.stream({"messages": st.session_state["messages"]}, config, stream_mode="values")
    for event in events:
        st.session_state["messages"] = event["messages"]

# # Display conversation
# for msg_type, msg_text in st.session_state["messages"]:
#     if msg_type == "user":
#         st.text_area("You", value=msg_text, height=100, key=f"user_{msg_text}", disabled=True)
#     else:
#         st.text_area("Bot", value=msg_text, height=100, key=f"bot_{msg_text}", disabled=True)
        # Iterate through the messages safely

# Iterate through the messages safely
for idx, message in enumerate(st.session_state["messages"]):
    if isinstance(message, HumanMessage):
        msg_type = "user"
        msg_text = message.content
    elif isinstance(message, AIMessage):
        msg_type = "bot"
        msg_text = message.content
    elif isinstance(message, ToolMessage):
        msg_type = "tool"
        msg_text = f"Tool {message.content}: {message.content}"
    else:
        st.error(f"Unexpected message type: {type(message)}")
        continue

    # Unique keys based on the index of the message
    if msg_type == "user":
        with st.chat_message("human"):
            st.markdown(msg_text)
    elif msg_type == "bot" and msg_text != '':
        # st.snow()
        # st.feedback(key=f"user_{idx}")
        with st.container(border=True):
            with st.chat_message("ai"):
                st.markdown(msg_text)
    elif msg_type == "tool":
        with st.expander(":red[Tool call]:toolbox::"):
            st.text_area("Tool Response", value=msg_text, height=300,
                          key=f"tool_{idx}", disabled=True,
                          label_visibility="collapsed")



