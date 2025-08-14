from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

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


def retrieve(state: State):
    retrieved_docs = vectorstore_client.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    doc_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": doc_content})
    response = llm.invoke(messages)
    return {"answer": response.content}



graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# for chunk, metadata in graph.stream(
#     {"question": "Summarize the Ramcharitmanas"}, stream_mode="messages"
# ):
#     if isinstance(chunk, AIMessage):
#         print(chunk.content, end="")
#     # print(f"{other}\n\n----------------\n")
# #
response = graph.invoke({"question": "Summarize the Ramcharitmanas."})
print(response["answer"])
