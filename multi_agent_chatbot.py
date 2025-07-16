from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [arxiv_tool, wikipedia_tool]


class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

graphbuilder = StateGraph(State)

llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.4)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": llm_with_tools.invoke(state["messages"])}

graphbuilder.add_node("chatbot", chatbot)
graphbuilder.add_edge(START, "chatbot")

tool_node = ToolNode(tools=tools)
graphbuilder.add_node("tools", tool_node)
graphbuilder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graphbuilder.add_edge("tools", "chatbot")
graphbuilder.add_edge("chatbot", END)

graph = graphbuilder.compile()

# from PIL import Image as PILImage
# from io import BytesIO

# try:
#     # Generate and display the graph
#     graph_image = graph.get_graph().draw_mermaid_png()
#     img = PILImage.open(BytesIO(graph_image))
#     img.show()  # This will open in your default image viewer
# except Exception as e:
#     print(f"Could not display graph: {e}")
'''
user_input = input("Enter your query: ")
events = graph.stream({"messages": [{"role": "user", "content": user_input}]})
for event in events:
    print(event['messages'][-1].pretty_print())
'''

user_input = input("Enter your query: ")
events = graph.stream({"messages": [{"role": "user", "content": user_input}]})

for event in events:
    for node_name, node_output in event.items():
        if node_name == "chatbot":
            # The messages field contains the actual message object
            message = node_output["messages"]
            print(f"Bot: {message.content}")
        elif node_name == "tools":
            print("🔧 Using tools...")









