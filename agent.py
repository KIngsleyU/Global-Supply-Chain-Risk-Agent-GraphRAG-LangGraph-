# LangGraph Logic (The Decision Maker)

"""
My plan implementation for this file

You correctly identified the flow: Start (Input) → The Messy Middle (Intermediate Steps) → The End (Output).

In LangGraph code, we typically implement this slightly differently to make it easier to manage. Instead of a key literally named intermediate_steps, we usually use a key called messages.

This list holds the entire conversation history: User input, AI reasoning, Tool calls, Tool outputs, and the Final Answer.

We use a special "reducer" called add_messages so that when a node returns data, it appends to this list rather than deleting the old history.

💻 Step 5: The Agent Logic (agent.py)

Let's start writing agent.py.

We need to:

Define the AgentState.

Initialize our Tools (the Graph search and Vector search functions we built earlier).

Bind the LLM to these tools.

The Challenge:

Can you write the code to imports the necessary libraries and define the AgentState class?

Hint: You will need TypedDict and Annotated from typing.

Hint: You will need add_messages from langgraph.graph.message.

Give it a try!
"""


from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from graph_ops import SupplyChainGraph
from vector_ops import ProductVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import networkx as nx
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os
load_dotenv()

# Initialize our specific instances
graph_db = SupplyChainGraph()
graph_db.generate_data() # Don't forget to generate the world!
vector_db = ProductVectorStore()
vector_db.add_products(graph_db.products) # Don't forget to index the world!

# 1. Vector Search Tool
def retrieve_product_info(query: str):
    """
    Useful to find products based on a semantic query like 'medical supplies'.
    query: str
    return: list[Product]
    
    example:
    products = retrieve_product_info("tech supplies")
    return products
    """
    # Call vector_db.get_products...
    return vector_db.get_products(query)

# 2. Graph Traversal Tool
def explore_graph_connections(node_name: str):
    """
    Useful to find what is connected to a specific node (e.g., finding suppliers at a location).    
    node_name: str
    return: list[str]
    
    example:
    suppliers = explore_graph_connections("Port of Shanghai")
    return suppliers
    """
    # NetworkX logic:
    # 1. Find the node object in graph_db.graph.nodes where name == node_name
    # 2. Return its neighbors (successors/predecessors)
    # Hint: nx.all_neighbors(graph_db.graph, found_node)
    found_node = graph_db.get_node_by_name(node_name)
    if not found_node:
        return []
    return list(nx.all_neighbors(graph_db.graph, found_node))

# Define the AgentState class: the State (The "Memory")
# This is the memory of the agent. It is a list of messages.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
# the Construction (The "Brain")
"""
    For the LLM, we use ChatOpenAI. 
    You also need the pre-built ToolNode from LangGraph to 
    actually run the tools.
"""

# ... Define tools list ...
tools = [retrieve_product_info, explore_graph_connections]


# LLM Setup
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter. Get your free key from https://openrouter.ai/")

model_name = "openai/gpt-oss-120b:free"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model=model_name,  # Default: Gemini Flash (supports tool calling)
    temperature=0
)

# ... Bind tools to LLM ...
llm_with_tools = llm.bind_tools(tools)

# ... Define the Nodes ...
def chatbot(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ... Build the Graph ...
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools)) # LangGraph's pre-built node to run functions

# ... Add Edges ...
builder.add_edge("__start__", "chatbot")
builder.add_conditional_edges("chatbot", tools_condition) # If tool call -> go to tools
builder.add_edge("tools", "chatbot") # Loop back to chatbot

agent = builder.compile()