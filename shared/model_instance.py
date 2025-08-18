from langchain_ollama.chat_models import ChatOllama

def get_model():
    return ChatOllama(model="llama3.1:8b", temperature=0.1,)
