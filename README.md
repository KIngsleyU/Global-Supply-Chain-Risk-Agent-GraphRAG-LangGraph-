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
├── agent.py           # 🚧 LangGraph Logic (Decision Making) - PLACEHOLDER
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
- `generate_locations()` - Generate 20-30 diverse supply chain locations (ports, warehouses, manufacturing facilities)
- `generate_suppliers(locations)` - Generate 1-3 suppliers per location
- `generate_products(suppliers)` - Generate 1-3 products per supplier
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

Implements the `ProductVectorStore` class for semantic product search using ChromaDB:

**Key Features:**
- Uses ChromaDB `EphemeralClient()` for in-memory operation (consistent with NetworkX approach)
- Indexes products by their names for semantic search
- Enables natural language queries to find relevant products
- Stores SKU and price in metadata for retrieval

**API:**
- `ProductVectorStore()` - Initialize an in-memory vector store
- `add_products(products)` - Index a list of Product objects
- `get_products(query, k)` - Semantic search for products matching a query (default: top 5 results)
- `get_all_products(ids)` - Retrieve products by their SKU IDs
- `delete_all_products(ids)` - Remove products from the collection by SKU

**Design Decisions:**
- Product names are embedded (semantically meaningful text like "Surgical Mask", "Hydraulic Pump")
- SKUs are stored in metadata (numeric identifiers with no semantic meaning)
- In-memory storage matches project philosophy of running entirely in RAM

**Example Usage:**
```python
from vector_ops import ProductVectorStore
from graph_ops import SupplyChainGraph

# Create vector store
vector_store = ProductVectorStore()

# Generate graph and products
graph = SupplyChainGraph()
graph.generate_data()

# Index products for semantic search
vector_store.add_products(graph.get_products())
# Output: "Indexed X products in ChromaDB."

# Search for products semantically
results = vector_store.get_products("medical equipment", k=5)
# Returns top 5 products semantically similar to "medical equipment"

# Get specific products by SKU
products = vector_store.get_all_products(["1234", "5678"])
```

### agent.py

*Placeholder for LangGraph agent implementation - orchestration and decision-making logic*

### main.py

Working entry point that demonstrates the integration of graph operations and vector search:

- Creates a `SupplyChainGraph` instance and generates synthetic supply chain data
- Creates a `ProductVectorStore` instance and indexes the generated products
- Demonstrates semantic search by querying for "medical equipment"
- Shows the complete workflow from graph generation to vector search

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

### Dependencies

- `networkx` - Graph database simulation and algorithms
- `chromadb` - Vector database for semantic search
- `faker` - Synthetic data generation for testing
- `langchain` - LLM framework integration
- `langgraph` - Agent orchestration and state management
- `matplotlib` - Graph visualization
- `scipy` - Scientific computing utilities

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
- `ProductVectorStore` class fully implemented in `vector_ops.py` with module-level docstring
- ChromaDB EphemeralClient integration for in-memory vector storage
- Product indexing by name for semantic search
- Semantic search functionality: `get_products(query, k)` finds products by natural language
- Direct product retrieval: `get_all_products(ids)` retrieves by SKU
- Product deletion support: `delete_all_products(ids)`
- Metadata storage (SKU, price) alongside embeddings

### 🚧 Phase 4: Agent Logic - PLANNED
- `agent.py` - Placeholder (LangGraph agent)
- Decision-making and orchestration logic needed
- Risk report generation workflow pending

### ✅ Phase 5: Entry Point - COMPLETE
- `main.py` - Working entry point demonstrating full workflow
- Integrates graph generation and vector indexing
- Demonstrates semantic search functionality
- No longer has duplicate code - uses proper imports

## Development Roadmap

- [x] Graph infrastructure and data generation (Phase 2)
- [x] Complete vector operations implementation (ChromaDB) - Phase 3
- [x] Working entry point demonstrating integration (Phase 5)
- [ ] Implement LangGraph agent with risk assessment logic (Phase 4)
- [ ] Add graph traversal methods for risk analysis:
  - Find suppliers by location
  - Find products by supplier
  - Calculate risk propagation through supply chain
- [ ] Implement risk report generation
- [ ] Add unit tests for graph operations and data generation
- [ ] Create example usage scripts demonstrating risk analysis workflows
- [ ] Add graph visualization capabilities using matplotlib/NetworkX
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
from vector_ops import ProductVectorStore

# Initialize components
graph = SupplyChainGraph()
vector_store = ProductVectorStore()

# Generate supply chain data
graph.generate_data()
# Output: "Generated 25 locations, 48 suppliers, 92 products"

# Index products for semantic search
products = graph.get_products()
vector_store.add_products(products)
# Output: "Indexed 92 products in ChromaDB."

# Perform semantic search
results = vector_store.get_products("medical equipment", k=5)
for product in results:
    print(f"{product.name} (SKU: {product.sku}, Price: ${product.price})")
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
