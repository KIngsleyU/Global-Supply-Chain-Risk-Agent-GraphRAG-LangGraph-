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
├── schema.py          # Data models (Supplier, Product, Location)
├── graph_ops.py       # NetworkX graph and synthetic data generation
├── graph_viz.py       # Graphviz builder shared by Streamlit and the agent
├── vector_ops.py      # ChromaDB vector stores (product, supplier, location)
├── agent.py           # LangGraph agent, tools, and optional Streamlit launcher
├── main.py            # CLI entry: launches risk map UI, then runs the agent
├── app.py             # Streamlit “Global Supply Chain Risk Map”
├── requirements.txt   # Python dependencies
└── README.md          # This file
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
- **Locations and suppliers**: Faker generates company names and facility/port-style location names
- **Products**: Category-based catalog (Electronics, Automotive, Pharmaceutical, Industrial, Packaging, Energy) with realistic item names, **prefixed SKUs** (e.g., category initials + number), and **category-specific price ranges** (not a single flat \$100–\$1,000 band)

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
- **Products**: 1-3 products per supplier; names combine a catalog line item with a numeric suffix; SKUs use a category prefix; prices are drawn from per-category min/max ranges
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

### graph_viz.py

Centralizes **Graphviz** rendering so the same layout is used from Streamlit and from the agent:

- **`build_supply_chain_graphviz(graph_db, risk_threshold=0.6)`** returns a `graphviz.Digraph` with:
  - Locations: blue boxes
  - Suppliers: red if `risk_score > threshold`, else green (risk shown in label when red)
  - Products: yellow ovals with name and price
  - Edges: `LOCATED_AT` (supplier → location); supplier → product edges are labeled **`MAKES`** in the diagram for readability (the underlying graph in `graph_ops` still uses edge type **`MANUFACTURES`**)

The module can be run as a script to print DOT source or render a PNG (`supply_chain_graph.png`) for offline inspection.

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
- Configured with cosine similarity for semantic search (`hnsw:space: cosine`)

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
- Cosine similarity configured for optimal semantic matching

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
- Retrieval tools provide fuzzy/semantic lookup; graph traversal uses exact names unless you resolve names via those tools first
- System message integration to guide LLM behavior and encourage final summaries
- **`launch_risk_map_ui(risk_threshold=0.6)`** (helper, not a LangGraph tool): serializes the **current in-memory** `graph_db` to a temporary **`supply_chain_graph.dot`** file, sets environment variables **`SC_GRAPH_DOT_PATH`** and **`SC_GRAPH_THRESHOLD`**, and starts **`streamlit run app.py`** as a subprocess so the browser shows the **same graph** as the agent process (avoids regenerating a second random world in the UI)

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

4. **`explore_graph_connections(node_name)`** - Graph neighbor lookup via NetworkX
   - Resolves the node with **`graph_db.get_node_by_name()`** (exact name match on supplier, location, or product)
   - If no exact match, returns guidance to use **`retrieve_location_info`**, **`retrieve_supplier_info`**, or **`retrieve_product_info`** for fuzzy resolution, then call again with the resolved name
   - Returns all **neighbors** (`nx.all_neighbors`) as strings when a node is found

**Architecture:**

- **AgentState**: TypedDict with `messages` field using `add_messages` reducer
- **Chatbot Node**: Calls LLM with tools bound, handles conversation flow, injects system messages when needed
- **Tools Node**: Uses LangGraph's pre-built `ToolNode` to execute function calls
- **Conditional Edges**: Routes between chatbot and tools based on tool call detection
- **Vector Stores**: Three separate vector stores initialized and indexed on module import

**LLM Configuration:**

- Uses OpenRouter API with configurable model via `OPENROUTER_MODEL` environment variable
- Default model: `z-ai/glm-4.5-air:free` (supports function calling/tool use)
- Requires `OPENROUTER_API_KEY` environment variable
- Temperature: 0 (deterministic responses)
- Max retries: 5
- System message guidance to ensure final summaries after tool calls

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
}, config={"recursion_limit": 50})

# Access all messages in conversation flow
for message in result["messages"]:
    if hasattr(message, 'content') and message.content:
        print(message.content)
    # Tool calls and results are also in the messages list
```

### main.py

Entry point module that orchestrates the LangGraph agent execution for supply chain risk analysis:

**Key Features:**

- Calls **`launch_risk_map_ui()`** from `agent.py` after printing the query, which opens the Streamlit risk map (default http://localhost:8501) using a DOT snapshot of the agent’s graph
- Initializes and invokes the LangGraph agent with user queries
- Displays comprehensive conversation flow including all messages, tool calls, and responses
- Handles errors gracefully with informative messages for different failure scenarios
- Automatic content deduplication to handle LLM response repetition
- Configurable recursion limit (default: 50) for complex multi-step agent reasoning
- Diagnostic output showing total messages, message types, and final status

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
- Final status summary with diagnostics

**Configuration:**

- Recursion limit: 50 (allows complex multi-step reasoning)
- Tokenizer parallelism disabled to prevent warnings

**Example Usage:**

```bash
# Run the agent with the predefined query (also launches the Streamlit risk map)
python main.py

# The script will:
# 1. Print the query and launch the Global Supply Chain Risk Map (Streamlit) if Graphviz/Streamlit are available
# 2. Load pre-initialized graph and vector stores and run the agent
# 3. Display the full conversation flow with tool calls
# 4. Show the final risk assessment (if the LLM provides one)
```

### app.py

Streamlit web application for interactive supply chain risk visualization:

**Key Features:**

- Interactive web-based visualization of the supply chain graph
- Risk threshold slider (used when **not** loading an agent snapshot) to highlight suppliers with risk scores above a configurable threshold
- Color-coded visualization:
  - Blue boxes: Locations
  - Red boxes: High-risk suppliers (risk_score > threshold)
  - Green boxes: Low-risk suppliers (risk_score <= threshold)
  - Yellow ovals: Products with pricing information
- Directed graph visualization showing relationships:
  - Supplier → LOCATED_AT → Location
  - Supplier → product edges (stored as **MANUFACTURES** in `graph_ops`; diagram edge label **MAKES** via `graph_viz.py`)

**Two rendering modes:**

1. **Standalone** (`streamlit run app.py` with no snapshot env): loads data with `@st.cache_resource` → `SupplyChainGraph().generate_data()`, builds the graph with `build_supply_chain_graphviz`, and the **sidebar slider** controls the threshold for coloring.
2. **Agent snapshot** (when **`SC_GRAPH_DOT_PATH`** points to an existing `.dot` file, as set by `launch_risk_map_ui`): reads the DOT string from disk and passes it to `st.graphviz_chart`. The visualization matches the agent’s in-memory graph at launch time; the caption shows the snapshot path.

**Architecture:**

- Uses Streamlit for web interface and user interaction
- Uses Graphviz via Streamlit’s `graphviz_chart` for graph rendering
- Uses `graph_viz.build_supply_chain_graphviz` for non-snapshot mode
- Integrates with `SupplyChainGraph` from `graph_ops` for standalone data access
- Caches graph data using `@st.cache_resource` for performance (standalone mode only)

**Usage:**

```bash
# Standalone: generate its own world and use the sidebar threshold
streamlit run app.py

# From Python (same graph as agent): see launch_risk_map_ui in agent.py
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
   - Copy `.env.example` to `.env`
   - Add your OpenRouter API key (get a free key from https://openrouter.ai/):

   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   # Optional: override the chat model (default in code: z-ai/glm-4.5-air:free)
   # OPENROUTER_MODEL=deepseek/deepseek-chat-v3:free
   ```

### Dependencies

- `networkx` - In-memory directed graph and traversal
- `chromadb` - Vector database for semantic search (Ephemeral client)
- `faker` - Synthetic company names, locations, and product SKU suffixes
- `langchain` - LLM framework integration
- `langchain-openai` - OpenAI-compatible client for OpenRouter
- `langgraph` - Agent orchestration and state management
- `python-dotenv` - Load `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL` from `.env`
- `graphviz` - Python package used with `graph_viz.py` and Streamlit’s `graphviz_chart`
- `streamlit` - Web UI for the risk map
- `matplotlib` / `scipy` - Listed in `requirements.txt` (not imported by the core modules in this repo; may satisfy transitive needs or local experiments)

**System requirement:** the Graphviz **engine** (`dot`) must be installed on your machine for Streamlit to render charts (e.g., `brew install graphviz` on macOS).

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
- Cosine similarity configuration (`hnsw:space: cosine`) for optimal semantic matching

### ✅ Phase 4: Agent Logic - COMPLETE

- `agent.py` - Fully implemented LangGraph agent with module-level docstring
- LangGraph `StateGraph` with `AgentState` and message-based conversation flow
- `launch_risk_map_ui()` integrates Streamlit with the same `graph_db` instance via a temporary DOT file
- Four tools implemented:
  - `retrieve_product_info(query)` - Semantic product search via ProductVectorStore
  - `retrieve_supplier_info(query)` - Semantic supplier search via SupplierVectorStore
  - `retrieve_location_info(query)` - Semantic location search via LocationVectorStore
  - `explore_graph_connections(node_name)` - Neighbor traversal after exact name match
- Uses LangGraph's pre-built `ToolNode` and `tools_condition` for tool execution
- Integrated with OpenRouter API (default: `z-ai/glm-4.5-air:free`, configurable via `OPENROUTER_MODEL`)
- Auto-initialization of graph and all three vector stores on module import
- `explore_graph_connections()` uses exact node names; misspellings are handled by the three `retrieve_*` tools
- System message integration to guide LLM behavior and encourage final summaries after tool calls
- Exact match detection in retrieval tools for optimal performance

### ✅ Phase 5: Entry Point - COMPLETE

- `main.py` - Working entry point with module-level docstring
- Orchestrates LangGraph agent execution with comprehensive error handling
- Displays full conversation flow including tool calls and responses
- Automatic content deduplication to handle LLM response repetition
- Error handling for API failures, recursion limits, and model compatibility issues
- Configurable recursion limit (50) for complex agent reasoning
- Diagnostic output for troubleshooting agent behavior

### ✅ Phase 6: Web Visualization - COMPLETE

- `graph_viz.py` - Shared Graphviz builder (`build_supply_chain_graphviz`) for Streamlit and the agent
- `app.py` - Streamlit “Global Supply Chain Risk Map” with standalone (cached `generate_data`) or **DOT snapshot** mode via `SC_GRAPH_DOT_PATH`
- `agent.py` - `launch_risk_map_ui()` writes a temp `.dot` file and spawns Streamlit so the UI matches the agent’s graph
- Interactive graph visualization using Graphviz (`st.graphviz_chart`)
- Risk threshold slider for dynamic filtering in standalone mode
- Color-coded visualization (locations, suppliers, products)
- Graph data caching for performance (`@st.cache_resource`) in standalone mode

## Development Roadmap

- [X] Graph infrastructure and data generation (Phase 2)
- [X] Complete vector operations implementation with three vector stores (Phase 3)
- [X] Implement LangGraph agent with four tools and semantic matching (Phase 4)
- [X] Working entry point with error handling and diagnostics (Phase 5)
- [X] Streamlit web application for graph visualization (Phase 6)
- [X] Basic graph traversal methods (`explore_graph_connections`, `get_node_by_name`)
- [X] Semantic matching via retrieval tools for fuzzy entity lookups
- [X] Three vector stores for products, suppliers, and locations
- [X] Exact match detection in retrieval tools for optimal performance
- [X] System message integration for LLM guidance
- [ ] Advanced risk analysis methods:
  - Calculate risk propagation through supply chain
  - Aggregate risk scores across affected products
- [ ] Implement risk report generation with formatted output
- [ ] Add unit tests for graph operations, vector stores, and data generation
- [ ] Create example usage scripts demonstrating risk analysis workflows
- [ ] Performance optimization for large-scale supply chain graphs
- [ ] Add query methods for finding affected entities during risk events

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

# Perform semantic search with fuzzy matching
products = product_store.get_products("medical equipment", k=5)
suppliers = supplier_store.get_suppliers("Acme Corporation", k=3)
locations = location_store.get_locations("Port of mexico city", k=3)
# Handles misspellings and partial matches automatically
```

### Example: Using the LangGraph Agent

```python
from agent import agent
from langchain_core.messages import HumanMessage

# Agent automatically initializes with graph and all three vector stores
# Retrieval tools handle fuzzy queries; explore_graph_connections needs exact names
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

**Note**: The **`retrieve_*`** tools use semantic search, so natural-language or misspelled queries can still surface the right entities. **`explore_graph_connections`** expects an exact graph node name (use a `retrieve_*` tool first if you only have a fuzzy description).

### Example: Using the Streamlit Web App

```bash
# Standalone: the app generates and caches its own supply chain world
streamlit run app.py

# The app will:
# 1. Load supply chain data (cached after first run via @st.cache_resource)
# 2. Display the graph; use the sidebar slider to change the risk threshold
# 3. Show suppliers above threshold in red

# To see the same graph as the agent (from Python):
#   from agent import launch_risk_map_ui
#   print(launch_risk_map_ui(0.6))
# Then open http://localhost:8501 — rendering uses the DOT snapshot from agent.py
```

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
