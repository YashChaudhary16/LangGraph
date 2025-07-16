#!/usr/bin/env python3
"""
Interactive LangGraph Chatbot with Groq LLM
Allows for continuous conversation
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

def chat_loop():
    """Interactive chat loop"""
    print("🤖 LangGraph Chatbot is ready!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")
    
    # Initialize conversation history
    conversation_state = {"messages": []}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation_state["messages"].append({"role": "user", "content": user_input})
            
            # Get response from chatbot
            result = graph.invoke(conversation_state)
            
            # Update conversation state
            conversation_state = result
            
            # Display bot response
            bot_response = result["messages"][-1].content
            print(f"Bot: {bot_response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    chat_loop() 