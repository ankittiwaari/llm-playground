from dotenv import load_dotenv
from shared.model_instance import get_model
from langchain_tavily import TavilySearch
from shared.config import DB_URI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from os import getenv
load_dotenv()


with PostgresSaver.from_conn_string(DB_URI) as memory:
    memory.setup()
    model = get_model()
    search = TavilySearch(max_results=2, tavily_api_key=getenv("TAVILY_API_KEY"))
    tools = [search]
    model_with_tools = model.bind_tools(tools)
    agent_executer = create_react_agent(model, tools)
    config = {"configurable": {"thread_id": "abc1234"}}
    input_message = {
        "role": "user",
        "content": "Hi, I am Bob and I live in Raebareli. search the weather where I live"
    }
    response = agent_executer.invoke({"messages": [input_message]})
    for step, metadata in agent_executer.stream(
            {"messages": [input_message]}, stream_mode="messages"
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            print(text, end="")
