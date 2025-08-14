from langchain_ollama.chat_models import ChatOllama

def get_model():
    return ChatOllama(model="gemma3:4b", temperature=0.1,)
