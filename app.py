import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
from datetime import datetime, UTC
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel     # we can mantain the structures of the questions of the input 



# Load environment variables
load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["chat"]
collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # all the origins can access the API
    allow_headers=["*"],       # allow all headers
    allow_credentials=True,      # allow cookies and authentication
)


# Prompt with proper history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study bot assistant. Remember what the user tells you and answer the questions based on the previous conversations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b"
)

chain = prompt | llm

# Fetch chat history from MongoDB
def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    messages = []

    for chat in chats:
        if chat["role"] == "user":
            messages.append(HumanMessage(content=chat["message"]))
        else:
            messages.append(AIMessage(content=chat["message"]))

    return messages


@app.get("/")     # root route
def home():
    return {"message": "Welcome to the Chat API!"}


@app.post("/chat")     # to post the  information from website to server
def chat(request: ChatRequest):
    history = get_history(request.user_id)
    response = chain.invoke({
        "history": history,
        "question": request.question
    })

    # Store user message
    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.now(UTC)
    })

    # Store assistant message
    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.now(UTC)
    })
    return {"response": response.content}
