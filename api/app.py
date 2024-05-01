from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# os.environ[]=os.getenv()

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai"
# )

llm1=Ollama(model="llama2")
llm2=Ollama(model="gemma:7b")
prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words. also tell me the name of your model")
prompt2=ChatPromptTemplate.from_template("Write me a poem about {topic}.The end word of each line must rhyme with the end word of the previous line.")

add_routes(
    app,
    prompt1|llm1,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm2,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

