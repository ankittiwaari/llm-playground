from langchain_core.messages import HumanMessage, SystemMessage
from shared.model_instance import get_model

model = get_model()

messages = [
    SystemMessage("Translate this from English to Hindi"),
    HumanMessage("How are you?")
]

response = model.invoke(messages)
print(response.content)