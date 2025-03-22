import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from interface import send_prompt, generate_history, load_point, load_history, clear_history
from store import load_vectorstore
from Chat import ChatMemoryModel

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = load_vectorstore()
print("Vector store loaded")

chat_model = ChatMemoryModel(model_name="llama3.1")

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

class HistoryItem(BaseModel):
    role: str
    content: str

class History(BaseModel):
    history: List[HistoryItem]

@app.post("/query", response_model=Answer)
async def process_query(query: Query):
    try:
        result = send_prompt(query.question, vectorstore, chat_model=chat_model)
        return Answer(answer=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=History)
async def get_history():
    history = generate_history(chat_model)
    return History(history=[HistoryItem(role=item["role"], content=item["content"]) for item in history])

@app.post("/load_point/{point}")
async def load_history_point(point: int):
    result = load_point(chat_model, point)
    return {"message": result}

@app.post("/load_history")
async def load_chat_history(file_path: str):
    result = load_history(chat_model, file_path)
    return {"message": result}

@app.post("/clear_history")
async def clear_chat_history():
    clear_history(chat_model)
    return {"message": "History cleared successfully"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)