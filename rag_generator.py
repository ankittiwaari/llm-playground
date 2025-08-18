from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from typing_extensions import TypedDict, List

from shared.config import DB_URI
from shared.model_instance import get_model
from shared.vectorstore_instance import get_vectorstore_client

vectorstore_client = get_vectorstore_client()
llm = get_model()

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always recheck the context before answering and refer to the context in your answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""
    retrieved_docs = vectorstore_client.similarity_search(query=query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n Content: {doc.page_content}" for doc in retrieved_docs)
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    llm_response = llm_with_tools.invoke(state['messages'])
    return {"messages": [llm_response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state['messages']):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]  # Get the last two tool messages
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "Always cite sources using their metadata (e.g., [Source: ...]). "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state['messages']
        if message.type in ("human", "system",)
           or (message.type == "ai" and not message.tool_calls)
    ]

    llm_prompt = [SystemMessage(system_message_content)] + conversation_messages

    llm_response = llm.invoke(llm_prompt)
    return {
        "messages": [llm_response],
        "sources": [d.metadata for msg in tool_messages for d in msg.artifact or []]
    }


graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)


def ask_llm(question):
    with PostgresSaver.from_conn_string(DB_URI) as memory:
        memory.setup()
        graph = graph_builder.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "abc123"}}
        input_message = question
        state = graph.invoke(
                {"messages": [{"role": "user", "content": input_message}]},
                config=config,
        )

        # Extract final AI answer
        answer = None
        if "messages" in state and state["messages"]:
            answer = state["messages"][-1].content

        # Extract sources if available (from generate node, option C earlier)
        sources = state.get("sources", [])

        return {"answer": answer, "sources": sources}
