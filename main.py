from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from rag_generator import ask_llm

class Question(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask")
def ask(q: Question):
    answer = ask_llm(q.question)
    return {
        "question": f"You asked: {q.question}",
        "answer": answer
    }