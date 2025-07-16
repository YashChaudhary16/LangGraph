#!/usr/bin/env python3
"""
Simple LangGraph Chatbot with Groq LLM
Minimal implementation without visualization
"""

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv(dotenv_path='.env', override=True)

# Define the state structure
class State(TypedDict):
    """
    Messages have a type of list and 
    add_messages is a function that adds messages to the list without overwriting existing messages
    """
    messages: Annotated[list[dict], add_messages]

# Initialize the LLM
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# Define the chatbot function
def chatbot(state: State) -> State:
    """Process messages through the LLM"""
    return {"messages": llm.invoke(state["messages"])}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Test the chatbot
if __name__ == "__main__":
    # Example usage
    test_messages = [{"role": "user", "content": "Hello! How are you?"}]
    
    print("Testing the chatbot...")
    print("Input:", test_messages[0]["content"])
    
    # Run the graph
    result = graph.invoke({"messages": test_messages})
    
    print("Output:", result["messages"][-1].content)
    print("Chatbot is working successfully!") 