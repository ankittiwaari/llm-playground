from langchain_ollama.chat_models import ChatOllama

def get_model():
    return ChatOllama(model="llama3", temperature=0.1,)
