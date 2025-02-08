from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API Keys: Check your .env file")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index
index_name = "medicalbot"

# Embed and upsert embeddings into Pinecone
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize OpenAI LLM with API key
llm = OpenAI(temperature=0.4, max_tokens=500, api_key=OPENAI_API_KEY)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create Retrieval-Augmented Generation (RAG) chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
