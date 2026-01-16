# Global Supply Chain Risk Agent - GraphRAG & LangGraph

An autonomous agent that monitors global supply chains and assesses risk using GraphRAG (Graph Retrieval-Augmented Generation) with LangGraph. The agent traverses knowledge graphs to identify downstream impacts when risk events occur in supply chains.

## Overview

This project implements a **Global Supply Chain Risk Guardian** agent. When a "Risk Event" (like a port strike, hurricane, or geopolitical disruption) occurs, the agent:

1. Searches the knowledge graph to find suppliers in the affected region
2. Traverses the graph to identify which products those suppliers manufacture
3. Calculates downstream impact on customers
4. Generates comprehensive risk reports

**Why GraphRAG?** A standard vector search would only return documents about "strikes" without understanding the actual relationships. GraphRAG enables proper traversal: `Location → Supplier → Product`, providing actionable insights about who is actually affected.

## Architecture

### Graph Schema

The knowledge graph consists of three node types:

- **Supplier**: Represents manufacturing companies with risk attributes (`name`, `risk_score`, `revenue`)
- **Product**: Represents manufactured goods (`name`, `sku`, `price`)
- **Location**: Represents geographical regions/cities (`name`, `country`)

#### Relationships

- `(Supplier)-[:MANUFACTURES]->(Product)` - Links suppliers to the products they manufacture
- `(Supplier)-[:LOCATED_AT]->(Location)` - Links suppliers to their physical locations

### Implementation Approach

The project uses **NetworkX** (in-memory directed graph) to simulate a graph database without requiring external infrastructure like Neo4j. This approach runs entirely in RAM, eliminating connection and Docker setup complexity while maintaining full graph traversal capabilities.

**Directed Graph Design**: The implementation uses `nx.DiGraph()` to ensure asymmetric relationships. For example, `Supplier → MANUFACTURES → Product` makes sense, but the reverse does not. This design choice prevents confusion in risk analysis during graph traversal.

## Project Structure

```
.
├── schema.py          # ✅ Data Models (Nodes) - IMPLEMENTED
├── graph_ops.py       # ✅ NetworkX Graph Operations - IMPLEMENTED
├── vector_ops.py      # ✅ ChromaDB Logic (Vector Search) - IMPLEMENTED
├── agent.py           # ✅ LangGraph Agent - IMPLEMENTED
├── main.py            # ✅ Entry Point - WORKING
├── requirements.txt   # ✅ Dependencies
└── README.md          # Project documentation
```

## Components

### schema.py

Defines the core data models using Python dataclasses:

```python
@dataclass
class Supplier:
    name: str
    risk_score: float
    revenue: float
    def __hash__(self):
        return hash(self.name)

@dataclass
class Product:
    name: str
    sku: str
    price: float

@dataclass
class Location:
    name: str
    country: str
```

**Key Features:**

- All three node classes implement `__hash__()` methods for NetworkX compatibility:
  - `Supplier`: Uses `name` as unique identifier
  - `Product`: Uses `sku` as unique identifier
  - `Location`: Uses `(name, country)` tuple for uniqueness
- Module-level docstring provides comprehensive documentation

### graph_ops.py

Implements the `SupplyChainGraph` class for managing the knowledge graph:

**Key Features:**

- Uses NetworkX `DiGraph()` for directed, asymmetric relationships
- Supports adding nodes (Supplier, Product, Location) and edges with type labels
- Provides comprehensive data generation methods for synthetic supply chain data
- Maintains internal state for locations, suppliers, and products
- Uses Faker library for realistic data generation

**API:**

- `SupplyChainGraph()` - Initialize an empty directed graph with state tracking
- `add_node(node)` - Add a node to the graph
- `add_edge(source, target, edge_type)` - Add a directed edge with a type label
- `get_graph()` - Retrieve the underlying NetworkX graph object
- `get_nodes()` - Get a list of all nodes in the graph
- `get_node_by_name(name)` - Find a node by its name attribute (returns node object or None)
- `generate_locations()` - Generate 20-30 diverse supply chain locations (ports, warehouses, manufacturing facilities)
- `generate_suppliers(locations)` - Generate 1-3 suppliers per location
- `generate_products(suppliers)` - Generate 1-3 products per supplier with unique SKUs
- `generate_data()` - Orchestrate full data generation pipeline
- `get_locations()` - Retrieve generated locations list
- `get_suppliers()` - Retrieve generated suppliers list
- `get_products()` - Retrieve generated products list

**Data Generation Details:**

- **Locations**: Generates 20-30 locations including ports (e.g., "Port of Shanghai"), warehouses (e.g., "Hamburg Warehouse"), and manufacturing facilities (e.g., "Shenzhen Manufacturing Facility")
- **Suppliers**: 1-3 suppliers per location with random risk scores (0-1) and revenue ($100K-$10M)
- **Products**: 1-3 products per supplier with unique SKUs and prices ($100-$1,000)
- All numeric values are rounded to 2 decimal places for consistency

**Example Usage:**

```python
from graph_ops import SupplyChainGraph

# Create graph instance
graph = SupplyChainGraph()

# Generate complete supply chain data
graph.generate_data()
# Prints: "Generated X locations, Y suppliers, Z products"

# Access the graph or individual collections
nx_graph = graph.get_graph()
locations = graph.get_locations()
suppliers = graph.get_suppliers()
products = graph.get_products()

# Or manually add nodes/edges
from schema import Supplier, Product, Location

supplier = Supplier(name="Acme Corp", risk_score=0.3, revenue=1000000)
product = Product(name="Widget A", sku="WID-A001", price=99.99)
location = Location(name="Port of Los Angeles", country="United States")

graph.add_node(supplier)
graph.add_node(product)
graph.add_node(location)
graph.add_edge(supplier, location, edge_type="LOCATED_AT")
graph.add_edge(supplier, product, edge_type="MANUFACTURES")
```

### vector_ops.py

Implements three vector store classes for semantic search over supply chain entities using ChromaDB:

**Classes:**

1. **ProductVectorStore** - Semantic search for products
2. **SupplierVectorStore** - Semantic search for suppliers  
3. **LocationVectorStore** - Semantic search for locations

**Key Features:**

- Uses ChromaDB `EphemeralClient()` for in-memory operation (consistent with NetworkX approach)
- Indexes entity names for semantic similarity search
- Enables natural language queries with fuzzy matching for misspellings and partial matches
- Stores metadata (SKU, price, risk_score, revenue, country) alongside embeddings
- Handles exact match detection when query matches entity name exactly

**ProductVectorStore API:**

- `ProductVectorStore()` - Initialize an in-memory vector store for products
- `add_products(products)` - Index a list of Product objects
- `get_products(query, k)` - Semantic search for products matching a query (default: top 5 results)
- `get_all_products(ids)` - Retrieve products by their SKU IDs
- `delete_all_products(ids)` - Remove products from the collection by SKU

**SupplierVectorStore API:**

- `SupplierVectorStore()` - Initialize an in-memory vector store for suppliers
- `add_suppliers(suppliers)` - Index a list of Supplier objects
- `get_suppliers(query, k)` - Semantic search for suppliers matching a query (default: top 5 results)
- `get_all_suppliers(names)` - Retrieve suppliers by their names

**LocationVectorStore API:**

- `LocationVectorStore()` - Initialize an in-memory vector store for locations
- `add_locations(locations)` - Index a list of Location objects
- `get_locations(query, k)` - Semantic search for locations matching a query (default: top 5 results)
- `get_all_locations(ids)` - Retrieve locations by their unique IDs (name::country format)

**Design Decisions:**

- Entity names are embedded (semantically meaningful text like "Surgical Mask", "Port of Shanghai", "Acme Corp")
- Unique identifiers stored in metadata (SKU for products, name for suppliers, name::country for locations)
- Semantic search enables fuzzy matching for misspellings (e.g., "Port of mexico city" → "Mexico City Manufacturing Facility")
- In-memory storage matches project philosophy of running entirely in RAM

**Example Usage:**

```python
from vector_ops import ProductVectorStore, SupplierVectorStore, LocationVectorStore
from graph_ops import SupplyChainGraph

# Create vector stores
product_store = ProductVectorStore()
supplier_store = SupplierVectorStore()
location_store = LocationVectorStore()

# Generate graph data
graph = SupplyChainGraph()
graph.generate_data()

# Index entities for semantic search
product_store.add_products(graph.get_products())
supplier_store.add_suppliers(graph.get_suppliers())
location_store.add_locations(graph.get_locations())

# Semantic search with fuzzy matching
products = product_store.get_products("medical equipment", k=5)
suppliers = supplier_store.get_suppliers("Acme Corporation", k=3)
locations = location_store.get_locations("Port of mexico city", k=3)
# Handles misspellings and partial matches automatically
```

### agent.py

Implements the LangGraph agent for supply chain risk analysis using graph traversal and semantic vector search:

**Key Features:**

- Uses LangGraph's `StateGraph` with `AgentState` for message-based conversation flow
- Integrates with OpenRouter API for free LLM access (configurable model, default: `z-ai/glm-4.5-air:free`)
- Uses LangGraph's pre-built `ToolNode` and `tools_condition` for tool execution
- Auto-initializes graph and all three vector stores (product, supplier, location) with synthetic data on import
- Semantic matching fallback for fuzzy entity lookups (handles misspellings automatically)

**Tools Implemented:**

1. **`retrieve_product_info(query)`** - Semantic product search using ProductVectorStore
   - Handles exact matches (returns single result if query matches product name exactly)
   - Falls back to semantic similarity search for partial matches

2. **`retrieve_supplier_info(query)`** - Semantic supplier search using SupplierVectorStore
   - Fuzzy matching for supplier names with misspellings or variations
   - Returns top 5 similar suppliers if exact match not found

3. **`retrieve_location_info(query)`** - Semantic location search using LocationVectorStore
   - Handles location name variations (e.g., "Port of mexico city" → "Mexico City Manufacturing Facility")
   - Returns top 5 similar locations with country information

4. **`explore_graph_connections(node_name)`** - Graph traversal with semantic fallback
   - Tries exact match first using `graph_db.get_node_by_name()`
   - If not found, uses semantic search via vector stores (location → supplier → product)
   - Informs LLM when similar match is used instead of exact match
   - Returns connected nodes (neighbors) as strings for LLM readability

**Architecture:**

- **AgentState**: TypedDict with `messages` field using `add_messages` reducer
- **Chatbot Node**: Calls LLM with tools bound, handles conversation flow
- **Tools Node**: Uses LangGraph's pre-built `ToolNode` to execute function calls
- **Conditional Edges**: Routes between chatbot and tools based on tool call detection
- **Vector Stores**: Three separate vector stores initialized and indexed on module import

**LLM Configuration:**

- Uses OpenRouter API with configurable model via `OPENROUTER_MODEL` environment variable
- Default model: `z-ai/glm-4.5-air:free` (supports function calling/tool use)
- Requires `OPENROUTER_API_KEY` environment variable
- Temperature: 0 (deterministic responses)
- Max retries: 5

**Initialization:**

On module import, the agent automatically:
1. Generates synthetic supply chain data (20-30 locations, 50-100 suppliers, 100+ products)
2. Initializes three vector stores (ProductVectorStore, SupplierVectorStore, LocationVectorStore)
3. Indexes all entities in their respective vector stores for semantic search

**Example Usage:**

```python
from agent import agent
from langchain_core.messages import HumanMessage

# Agent is auto-initialized with data and vector stores
result = agent.invoke({
    "messages": [HumanMessage(content="Assess the impact of a strike at the Port of mexico city")]
})

# Access all messages in conversation flow
for message in result["messages"]:
    if hasattr(message, 'content') and message.content:
        print(message.content)
    # Tool calls and results are also in the messages list
```

### main.py

Entry point module that orchestrates the LangGraph agent execution for supply chain risk analysis:

**Key Features:**

- Initializes and invokes the LangGraph agent with user queries
- Displays comprehensive conversation flow including all messages, tool calls, and responses
- Handles errors gracefully with informative messages for different failure scenarios
- Automatic content deduplication to handle LLM response repetition
- Configurable recursion limit (default: 50) for complex multi-step agent reasoning

**Error Handling:**

- **ValueError (502/Upstream errors)**: OpenRouter service unavailability with helpful suggestions
- **GraphRecursionError**: Detects infinite loops, provides debugging guidance
- **BadRequestError (400)**: Model compatibility issues with alternative model recommendations
- **Generic Exception**: Catches and displays unexpected errors

**Output:**

- Prints the user query
- Displays the complete agent conversation flow:
  - Human messages (user queries)
  - AI messages (LLM responses and tool call decisions)
  - Tool messages (tool execution results)
  - Tool call details (function names and arguments)

**Configuration:**

- Recursion limit: 50 (allows complex multi-step reasoning)
- Tokenizer parallelism disabled to prevent warnings

**Example Usage:**

```bash
# Run the agent with the predefined query
python main.py

# The agent will:
# 1. Load pre-initialized graph and vector stores
# 2. Process the query about supply chain disruptions
# 3. Display the full conversation flow with tool calls
# 4. Show the final risk assessment
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env` (if it exists)
   - Add your OpenRouter API key (get a free key from https://openrouter.ai/):

   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   OPENROUTER_MODEL=z-ai/glm-4.5-air:free  # Optional: defaults to this model
   ```

   **Note**: The default model (`z-ai/glm-4.5-air:free`) supports function calling. Other free models that work include:
   - `deepseek/deepseek-chat-v3:free`
   - `qwen/qwen-2.5-72b-instruct:free`
   - `meta-llama/llama-3.3-70b-instruct:free`

### Dependencies

- `networkx` - Graph database simulation and algorithms
- `chromadb` - Vector database for semantic search
- `faker` - Synthetic data generation for testing
- `langchain` - LLM framework integration
- `langchain-openai` - OpenAI-compatible API integration for OpenRouter
- `langgraph` - Agent orchestration and state management
- `matplotlib` - Graph visualization
- `scipy` - Scientific computing utilities
- `python-dotenv` - Environment variable management from `.env` files

## Current Status

### ✅ Phase 1: Schema Blueprint - COMPLETE

- Graph schema defined (nodes and edges)
- Data models fully implemented in `schema.py` with module-level docstring
- All three node classes with proper type hints and hash methods:
  - `Supplier.__hash__()` uses name
  - `Product.__hash__()` uses SKU
  - `Location.__hash__()` uses (name, country) tuple

### ✅ Phase 2: Graph Infrastructure - COMPLETE

- `SupplyChainGraph` class implemented in `graph_ops.py`
- NetworkX directed graph container operational
- Node and edge addition methods functional
- Graph introspection methods available
- Synthetic data generation pipeline complete:
  - `generate_locations()` - Creates 20-30 diverse supply chain locations
  - `generate_suppliers()` - Creates 1-3 suppliers per location with realistic attributes
  - `generate_products()` - Creates 1-3 products per supplier
  - `generate_data()` - Orchestrates full data generation workflow
- Helper methods for accessing generated data (`get_locations()`, `get_suppliers()`, `get_products()`)

### ✅ Phase 3: Vector Operations - COMPLETE

- Three vector store classes fully implemented in `vector_ops.py` with module-level docstring:
  - `ProductVectorStore` - Semantic product search
  - `SupplierVectorStore` - Semantic supplier search with fuzzy matching
  - `LocationVectorStore` - Semantic location search with fuzzy matching
- ChromaDB EphemeralClient integration for in-memory vector storage
- Entity indexing by name for semantic search across all entity types
- Semantic search functionality with exact match detection:
  - `get_products(query, k)` - Finds products by natural language
  - `get_suppliers(query, k)` - Finds suppliers with misspelling tolerance
  - `get_locations(query, k)` - Finds locations with name variations
- Direct entity retrieval methods for all stores
- Metadata storage (SKU, price, risk_score, revenue, country) alongside embeddings
- Handles duplicate location entries using `set()` for unique indexing

### ✅ Phase 4: Agent Logic - COMPLETE

- `agent.py` - Fully implemented LangGraph agent with module-level docstring
- LangGraph `StateGraph` with `AgentState` and message-based conversation flow
- Four tools implemented:
  - `retrieve_product_info(query)` - Semantic product search via ProductVectorStore
  - `retrieve_supplier_info(query)` - Semantic supplier search via SupplierVectorStore
  - `retrieve_location_info(query)` - Semantic location search via LocationVectorStore
  - `explore_graph_connections(node_name)` - Graph traversal with semantic fallback
- Uses LangGraph's pre-built `ToolNode` and `tools_condition` for tool execution
- Integrated with OpenRouter API (default: `z-ai/glm-4.5-air:free`, configurable via `OPENROUTER_MODEL`)
- Auto-initialization of graph and all three vector stores on module import
- Semantic matching fallback in `explore_graph_connections()` handles misspellings automatically
- Exact match detection in retrieval tools for optimal performance

### ✅ Phase 5: Entry Point - COMPLETE

- `main.py` - Working entry point with module-level docstring
- Orchestrates LangGraph agent execution with comprehensive error handling
- Displays full conversation flow including tool calls and responses
- Automatic content deduplication to handle LLM response repetition
- Error handling for API failures, recursion limits, and model compatibility issues
- Configurable recursion limit (50) for complex agent reasoning

## Development Roadmap

- [X] Graph infrastructure and data generation (Phase 2)
- [X] Complete vector operations implementation with three vector stores (Phase 3)
- [X] Implement LangGraph agent with four tools and semantic matching (Phase 4)
- [X] Working entry point with error handling and deduplication (Phase 5)
- [X] Basic graph traversal methods (`explore_graph_connections`, `get_node_by_name`)
- [X] Semantic matching for fuzzy entity lookups (handles misspellings automatically)
- [X] Three vector stores for products, suppliers, and locations
- [X] Exact match detection in retrieval tools for optimal performance
- [ ] Advanced risk analysis methods:
  - Calculate risk propagation through supply chain
  - Aggregate risk scores across affected products
- [ ] Implement risk report generation with formatted output
- [ ] Add unit tests for graph operations, vector stores, and data generation
- [ ] Create example usage scripts demonstrating risk analysis workflows
- [ ] Add graph visualization capabilities using matplotlib/NetworkX
- [ ] Add query methods for finding affected entities during risk events
- [ ] Performance optimization for large-scale supply chain graphs

## Usage

### Basic Example: Generate Supply Chain Data

```python
from graph_ops import SupplyChainGraph

# Initialize the graph
graph = SupplyChainGraph()

# Generate complete supply chain network
graph.generate_data()
# Output: "Generated 25 locations, 48 suppliers, 92 products"

# Access the NetworkX graph for analysis
nx_graph = graph.get_graph()
print(f"Total nodes: {nx_graph.number_of_nodes()}")
print(f"Total edges: {nx_graph.number_of_edges()}")

# Access individual collections
locations = graph.get_locations()
suppliers = graph.get_suppliers()
products = graph.get_products()

# Example: Find suppliers at a specific location
for supplier in suppliers:
    for neighbor in nx_graph.neighbors(supplier):
        if isinstance(neighbor, Location):
            if neighbor.name == "Port of Shanghai":
                print(f"Supplier {supplier.name} is located at {neighbor.name}")
```

### Example: Complete Workflow (Graph + Vector Search)

```python
from graph_ops import SupplyChainGraph
from vector_ops import ProductVectorStore, SupplierVectorStore, LocationVectorStore

# Initialize components
graph = SupplyChainGraph()
product_store = ProductVectorStore()
supplier_store = SupplierVectorStore()
location_store = LocationVectorStore()

# Generate supply chain data
graph.generate_data()
# Output: "Generated 26 locations, 57 suppliers, 117 products"

# Index all entities for semantic search
product_store.add_products(graph.get_products())
supplier_store.add_suppliers(graph.get_suppliers())
location_store.add_locations(graph.get_locations())
# Output: "Indexed X products/suppliers/locations in ChromaDB."

# Perform semantic search with fuzzy matching
products = product_store.get_products("medical equipment", k=5)
suppliers = supplier_store.get_suppliers("Acme Corporation", k=3)
locations = location_store.get_locations("Port of mexico city", k=3)

# Handle exact matches (tools return single result) or similar matches
for product in products:
    print(f"{product.name} (SKU: {product.sku}, Price: ${product.price})")
```

### Example: Using the LangGraph Agent

```python
from agent import agent
from langchain_core.messages import HumanMessage

# Agent automatically initializes with graph and all three vector stores
# Query the agent (semantic matching handles misspellings automatically)
result = agent.invoke({
    "messages": [HumanMessage(content="Assess the impact of a strike at the Port of mexico city")]
}, config={"recursion_limit": 50})

# Extract and print all messages (includes tool calls and responses)
for message in result["messages"]:
    if hasattr(message, 'content') and message.content:
        print(message.content)
    # Tool calls are embedded in AIMessage objects
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Tool: {tool_call.get('name')} with args: {tool_call.get('args')}")
```

**Note**: The agent uses semantic matching, so queries like "Port of mexico city" will automatically find similar locations like "Mexico City Manufacturing Facility" even with misspellings or variations.

### Example: Manual Graph Construction

```python
from graph_ops import SupplyChainGraph
from schema import Supplier, Product, Location

graph = SupplyChainGraph()

# Create nodes
location = Location(name="Port of Rotterdam", country="Netherlands")
supplier = Supplier(name="Global Tech Industries", risk_score=0.45, revenue=5000000)
product = Product(name="Advanced Sensor Array", sku="ASA-2024-001", price=750.00)

# Add to graph
graph.add_node(location)
graph.add_node(supplier)
graph.add_node(product)

# Create relationships
graph.add_edge(supplier, location, edge_type="LOCATED_AT")
graph.add_edge(supplier, product, edge_type="MANUFACTURES")
```

## Contributing

*Contributing guidelines will be added as the project matures.*

## License

*License information to be added.*
