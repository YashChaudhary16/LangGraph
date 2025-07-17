from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)


# Define the tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

arxiv_wrapper = ArxivAPIWrapper()
wikipedia_wrapper = WikipediaAPIWrapper()

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [arxiv_tool, wikipedia_tool]


# Define the state

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

# Define the LLM and bind the tools

llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.4)
llm_with_tools = llm.bind_tools(tools)


# Define the chatbot node function

def chatbot(state: State) -> State:
    return {"messages": llm_with_tools.invoke(state["messages"])}


# Define the graph

graph_builder = StateGraph(State)

# Define the nodes

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

# Define the edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Take the user input and stream the response

user_input = input("Enter your query: ")
events = graph.stream({"messages": [{"role": "user", "content": user_input}]})

for event in events:
    # print(event)
    for node_name, node_output in event.items():
        if node_name == "chatbot":
            # The messages field contains the actual message object
            message = node_output["messages"]
            print(f"Bot: {message.content}")
        elif node_name == "tools":
            print("🔧 Using tools...")
