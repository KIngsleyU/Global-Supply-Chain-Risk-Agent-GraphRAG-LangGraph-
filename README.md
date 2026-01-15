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
├── vector_ops.py      # 🚧 ChromaDB Logic (Vector Search) - PLACEHOLDER
├── agent.py           # 🚧 LangGraph Logic (Decision Making) - PLACEHOLDER
├── main.py            # 📝 Entry Point - NEEDS REFACTORING
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

The `Supplier.__hash__()` method enables proper node identification in the graph structure.

### graph_ops.py

Implements the `SupplyChainGraph` class for managing the knowledge graph:

**Key Features:**
- Uses NetworkX `DiGraph()` for directed, asymmetric relationships
- Supports adding nodes (Supplier, Product, Location) and edges with type labels
- Provides methods for graph introspection

**API:**
- `SupplyChainGraph()` - Initialize an empty directed graph
- `add_node(node)` - Add a node to the graph
- `add_edge(source, target, edge_type)` - Add a directed edge with a type label
- `get_graph()` - Retrieve the underlying NetworkX graph object
- `get_nodes()` - Get a list of all nodes in the graph

**Example Usage:**
```python
from graph_ops import SupplyChainGraph
from schema import Supplier, Product, Location

graph = SupplyChainGraph()
supplier = Supplier("Acme Corp", risk_score=0.3, revenue=1000000)
product = Product("Widget A", sku="WID-A001", price=99.99)

graph.add_node(supplier)
graph.add_node(product)
graph.add_edge(supplier, product, edge_type="MANUFACTURES")
```

### vector_ops.py

*Placeholder for ChromaDB integration - semantic search functionality for risk event queries*

### agent.py

*Placeholder for LangGraph agent implementation - orchestration and decision-making logic*

### main.py

*Currently contains duplicate dataclass definitions - should import from `schema.py` instead*

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
- Data models fully implemented in `schema.py`
- All three node classes with proper type hints and hash methods

### ✅ Phase 2: Graph Infrastructure - COMPLETE
- `SupplyChainGraph` class implemented in `graph_ops.py`
- NetworkX directed graph container operational
- Node and edge addition methods functional
- Graph introspection methods available

### 🚧 Phase 3: Vector Operations - PLANNED
- `vector_ops.py` - Placeholder (ChromaDB integration)
- Semantic search functionality for risk events needed
- Vector embedding and retrieval logic pending

### 🚧 Phase 4: Agent Logic - PLANNED
- `agent.py` - Placeholder (LangGraph agent)
- Decision-making and orchestration logic needed
- Risk report generation workflow pending

### 📝 Phase 5: Entry Point - NEEDS REFACTORING
- `main.py` - Contains duplicate dataclass definitions
- Should import from `schema.py` to avoid code duplication
- Main execution flow and example usage needed

## Development Roadmap

- [ ] Complete vector operations implementation (ChromaDB)
- [ ] Implement LangGraph agent with risk assessment logic
- [ ] Refactor `main.py` to use proper imports
- [ ] Add graph traversal methods for risk analysis
- [ ] Implement risk report generation
- [ ] Add unit tests
- [ ] Create example usage scripts
- [ ] Add graph visualization capabilities

## Contributing

*Contributing guidelines will be added as the project matures.*

## License

*License information to be added.*
