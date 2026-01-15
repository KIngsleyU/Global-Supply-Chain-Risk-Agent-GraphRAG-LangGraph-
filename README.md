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

- **Supplier**: Represents manufacturing companies with risk attributes
- **Product**: Represents manufactured goods
- **Location**: Represents geographical regions/cities

#### Relationships

- `(Supplier)-[:MANUFACTURES]->(Product)`
- `(Supplier)-[:LOCATED_AT]->(Location)`

### Implementation Approach

I'm implementing this using **NetworkX** (in-memory graph) to simulate a graph database without requiring external infrastructure like Neo4j. This approach runs entirely in RAM, eliminating connection and Docker setup complexity while maintaining full graph traversal capabilities.

## Data Models

The core entities are defined as Python dataclasses in `schema.py`:

```python
@dataclass
class Supplier:
    name: str
    risk_score: float
    revenue: float
    # Unique ID helper for graph operations
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

The `risk_score` property on the `Supplier` node enables direct risk assessment calculations during graph traversal.

## Project Structure

```
.
├── schema.py          # ✅ Data Models (Nodes) - IMPLEMENTED
├── graph_ops.py       # 🚧 NetworkX Logic (Graph Operations) - PLACEHOLDER
├── vector_ops.py      # 🚧 ChromaDB Logic (Vector Search) - PLACEHOLDER
├── agent.py           # 🚧 LangGraph Logic (Decision Making) - PLACEHOLDER
├── main.py            # 📝 Entry Point - NEEDS REFACTORING
├── requirements.txt   # ✅ Dependencies
└── README.md          # Project documentation
```

### File Status

- **schema.py**: ✅ Fully implemented with all three node classes (`Supplier`, `Product`, `Location`)
- **graph_ops.py**: Contains only placeholder comment, NetworkX implementation needed
- **vector_ops.py**: Contains only placeholder comment, ChromaDB integration needed
- **agent.py**: Contains only placeholder comment, LangGraph agent implementation needed
- **main.py**: Currently contains duplicate dataclass definitions that should import from `schema.py`

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `networkx` - Graph database simulation
- `chromadb` - Vector database for semantic search
- `faker` - Synthetic data generation
- `langchain` - LLM framework
- `langgraph` - Agent orchestration
- `matplotlib` - Visualization
- `scipy` - Scientific computing

## Current Status

### ✅ Phase 1: Schema Blueprint - COMPLETE
- Graph schema defined (nodes and edges)
- Data models fully implemented in `schema.py`:
  - `Supplier` class with `name`, `risk_score`, `revenue`, and `__hash__` method
  - `Product` class with `name`, `sku`, `price`
  - `Location` class with `name`, `country`

### 🚧 Phase 2: Graph Infrastructure - PLANNED
- `graph_ops.py` - Placeholder (NetworkX graph operations)
- NetworkX graph container and traversal logic needed

### 🚧 Phase 3: Vector Operations - PLANNED
- `vector_ops.py` - Placeholder (ChromaDB integration)
- Semantic search functionality needed

### 🚧 Phase 4: Agent Logic - PLANNED
- `agent.py` - Placeholder (LangGraph agent)
- Decision-making and orchestration logic needed

### 📝 Phase 5: Entry Point - NEEDS REFACTORING
- `main.py` - Currently contains duplicate dataclass definitions
- Should import from `schema.py` instead of redefining classes

## Usage

*Usage instructions will be added once core functionality is implemented.*

## Development Notes

- `schema.py` is the source of truth for data models
- `main.py` needs to be refactored to import from `schema.py` to avoid code duplication

## License

[Your License Here]
