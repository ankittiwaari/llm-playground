from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from shared.model_instance import get_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from shared.config import DB_URI
model = get_model()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that only answers in {language}. Translate your answers to {language} and reply with translated answer.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages":trimmed_messages, "language": state["language"] }
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

config = {"configurable":{"thread_id":"abc123"}}

with PostgresSaver.from_conn_string(DB_URI) as memory:
    memory.setup()
    app = workflow.compile(checkpointer=memory)
    query = "What's my name?"
    language = "English"
    input_messages = [HumanMessage(query)]
    for chunk, metadata in app.stream(
        {"messages":input_messages, "language":language},
        config,
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="")