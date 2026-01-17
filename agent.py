# LangGraph Logic (The Decision Maker)

"""
LangGraph Agent Module for Supply Chain Risk Analysis

This module implements an autonomous agent using LangGraph that combines graph traversal
and vector search to analyze supply chain risks. The agent uses a message-based conversation
flow to answer natural language queries about supply chain entities and relationships.

Architecture:
- AgentState: TypedDict with messages field using add_messages reducer for conversation history
- Chatbot Node: Invokes LLM with bound tools to process user queries
- Tools Node: Executes function calls using LangGraph's pre-built ToolNode
- Conditional Routing: Routes between chatbot and tools based on tool call detection

Tools:
- retrieve_product_info(query): Semantic product search using ChromaDB vector store
- retrieve_supplier_info(query): Semantic supplier search with fuzzy matching
- retrieve_location_info(query): Semantic location search with fuzzy matching
- explore_graph_connections(node_name): Graph traversal to find connected nodes via NetworkX (uses semantic matching when exact match fails)

LLM Configuration:
- Uses OpenRouter API with GPT-OSS 120B free model
- Requires OPENROUTER_API_KEY environment variable
- Model: openai/gpt-oss-120b:free

Initialization:
- Automatically generates synthetic supply chain data on module import
- Initializes graph with 20-30 locations, suppliers, and products
- Indexes all products, suppliers, and locations in their respective vector stores for semantic search

Example:
    >>> from agent import agent
    >>> from langchain_core.messages import HumanMessage
    >>> 
    >>> result = agent.invoke({
    ...     "messages": [HumanMessage(content="What suppliers are in China?")]
    ... })
    >>> 
    >>> for message in result["messages"]:
    ...     if hasattr(message, 'content') and message.content:
    ...         print(message.content)
"""


from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from graph_ops import SupplyChainGraph
from vector_ops import ProductVectorStore, SupplierVectorStore, LocationVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import networkx as nx
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os

# Fix tokenizer warning about parallelism after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Initialize our specific instances
graph_db = SupplyChainGraph()
graph_db.generate_data() # Don't forget to generate the world!

# Initialize vector stores for semantic search
product_vector_db = ProductVectorStore()
supplier_vector_db = SupplierVectorStore()
location_vector_db = LocationVectorStore()

# Index all entities in their respective vector stores
product_vector_db.add_products(graph_db.products)
supplier_vector_db.add_suppliers(graph_db.suppliers)
location_vector_db.add_locations(graph_db.locations)

# 1. Product Search Tool
def retrieve_product_info(query: str):
    """
    Useful to find products based on a semantic query like 'medical supplies'.
    Uses semantic similarity search to handle misspellings and partial matches.
    
    query: str - Search query (e.g., "medical supplies", "tech products")
    return: list[str] - List of product descriptions as strings
    
    example:
    products = retrieve_product_info("tech supplies")
    return products
    """
    products = product_vector_db.get_products(query, k=5)
    return_products = [] # list to store the products
    for product in products:
        if query.lower() == product.name.lower():
            return [str(product)]
        return_products.append(str(product))  # add the product to the list 
    return return_products

# 2. Supplier Search Tool
def retrieve_supplier_info(query: str):
    """
    Useful to find suppliers based on a semantic query or name.
    Uses semantic similarity search to handle misspellings and partial matches.
    If exact match not found, returns top similar matches.
    
    query: str - Search query (e.g., "Acme Corp", "Mexican suppliers")
    return: list[str] - List of supplier descriptions as strings, or message if no exact match
    
    example:
    suppliers = retrieve_supplier_info("Acme Corporation")
    return suppliers
    """
    suppliers = supplier_vector_db.get_suppliers(query, k=5)
    if not suppliers:
        return [f"No suppliers found matching '{query}'. Try a different search term."]
    return_suppliers = [] # list to store the suppliers 
    for supplier in suppliers:
        if query.lower() == supplier.name.lower():
            return [str(supplier)]
        return_suppliers.append(str(supplier))  # add the supplier to the list
    return return_suppliers

# 3. Location Search Tool
def retrieve_location_info(query: str):
    """
    Useful to find locations based on a semantic query or name.
    Uses semantic similarity search to handle misspellings, partial matches, and variations.
    If exact match not found, returns top similar matches.
    
    query: str - Search query (e.g., "Port of Shanghai", "Mexico City", "Shanghai port")
    return: list[str] - List of location descriptions as strings, or message if no exact match
    
    example:
    locations = retrieve_location_info("Port of mexico city")
    # Will find "Mexico City Manufacturing Facility" or similar even with misspelling
    return locations
    """
    locations = location_vector_db.get_locations(query, k=5)
    if not locations:
        return [f"No locations found matching '{query}'. Try a different search term."]
    return_locations = [] # list to store the locations
    for location in locations:
        if query.lower() == location.name.lower():
            return [str(location)]
        return_locations.append(str(location))  # add the location to the list
    return return_locations
# 4. Graph Traversal Tool
def explore_graph_connections(node_name: str):
    """
    Useful to find what is connected to a specific node in the graph (e.g., finding suppliers at a location,
    products from a supplier, locations for a supplier).
    
    Uses semantic matching via vector stores when exact match fails. This handles misspellings,
    partial matches, and variations (e.g., "Port of mexico city" will find "Mexico City Manufacturing Facility").
    
    node_name: str - Name of the node to search for (can be supplier, location, or product name)
    return: list[str] - List of connected nodes as strings, or helpful message if not found
    
    example:
    suppliers = explore_graph_connections("Port of Shanghai")
    products = explore_graph_connections("Acme Corp")
    return suppliers or products
    """
    # Try exact match first
    found_node = graph_db.get_node_by_name(node_name)
    is_exact_match = found_node is not None
    matched_via_semantic = False
    
    # If not found, use semantic search via vector stores
    if not found_node:
        # Try location vector store first (most common use case)
        locations = location_vector_db.get_locations(node_name, k=1)
        if locations:
            found_node = graph_db.get_node_by_name(locations[0].name)
            matched_via_semantic = True
        
        # If still not found, try supplier vector store
        if not found_node:
            suppliers = supplier_vector_db.get_suppliers(node_name, k=1)
            if suppliers:
                found_node = graph_db.get_node_by_name(suppliers[0].name)
                matched_via_semantic = True
        
        # If still not found, try product vector store
        if not found_node:
            products = product_vector_db.get_products(node_name, k=1)
            if products:
                found_node = graph_db.get_node_by_name(products[0].name)
                matched_via_semantic = True
        
        # If still no match - suggest using retrieve_location_info or retrieve_supplier_info
        if not found_node:
            return [
                f"No node found matching '{node_name}' (exact or similar).\n"
                f"Try using retrieve_location_info('{node_name}') to find similar locations, "
                f"or retrieve_supplier_info('{node_name}') to find similar suppliers."
            ]
    
    neighbors = list(nx.all_neighbors(graph_db.graph, found_node))
    if not neighbors:
        return [f"Node '{found_node.name}' found but has no connections in the graph."]
    
    # Build result with connected nodes
    connected_nodes = [str(neighbor) for neighbor in neighbors]
    
    # If matched via semantic search, inform the LLM about the similarity match
    if matched_via_semantic:
        return [
            f"Note: No exact match for '{node_name}', but found a semantically similar node: {found_node.name}, and this semantically similar node has connections to the following nodes:\n"
            f"Connected nodes:\n" + "\n".join(connected_nodes)
        ]
    
    # Exact match - return just the connected nodes
    return connected_nodes

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
tools = [
    retrieve_product_info,
    retrieve_supplier_info,
    retrieve_location_info,
    explore_graph_connections
]


# LLM Setup
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter. Get your free key from https://openrouter.ai/")

# Model selection for OpenRouter
# Note: Not all free models support function calling/tool use properly
# Try these models that are known to work with function calling:
# - deepseek/deepseek-chat-v3:free (recommended - reliable)
# - qwen/qwen-2.5-72b-instruct:free
# - meta-llama/llama-3.3-70b-instruct:free
# model_name = "openai/gpt-oss-20b:free"
# model_name = "openai/gpt-oss-120b:free"
model_name = os.getenv(
    "OPENROUTER_MODEL", 
    # "google/gemini-2.0-flash-exp:free",
    "z-ai/glm-4.5-air:free")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model=model_name,  # Default: Gemini Flash (supports tool calling)
    temperature=0,
    max_retries=5
)

# ... Bind tools to LLM ...
llm_with_tools = llm.bind_tools(tools)

# ... Define the Nodes ...
def chatbot(state: AgentState):
    from langchain_core.messages import SystemMessage
    
    messages = state["messages"]
    
    # Count tool calls to detect when we're getting too deep without a summary
    tool_call_count = sum(1 for msg in messages if hasattr(msg, 'tool_calls') and msg.tool_calls)
    
    # Check if we have a system message already
    has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
    
    # Add system message on first call if not present
    if not has_system_message:
        system_msg = SystemMessage(
            content="You are a supply chain risk analysis agent. After gathering information with tools, "
                   "you MUST provide a clear summary response. Do not make endless tool calls - provide "
                   "an assessment after collecting sufficient data (usually after 5-8 tool calls)."
        )
        messages = [system_msg] + messages
    
    # If we've made many tool calls but no final response yet, add a reminder
    if tool_call_count >= 12:
        reminder = SystemMessage(
            content="IMPORTANT: You have made many tool calls. Stop now and provide your final assessment "
                   "and summary based on all the information you've collected. Do not make more tool calls."
        )
        messages = messages + [reminder]
    
    return {"messages": [llm_with_tools.invoke(messages)]}

# ... Build the Graph ...
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools)) # LangGraph's pre-built node to run functions

# ... Add Edges ...
builder.add_edge("__start__", "chatbot")
builder.add_conditional_edges("chatbot", tools_condition) # If tool call -> go to tools
builder.add_edge("tools", "chatbot") # Loop back to chatbot

agent = builder.compile()