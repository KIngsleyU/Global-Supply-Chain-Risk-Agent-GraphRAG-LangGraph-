# NetworkX Logic (The World)

"""
my plan implementation for this file

You are absolutely right to use nx.DiGraph() (Directed Graph). In our supply chain, relationships are asymmetric:

Supplier → MANUFACTURES → Product makes sense.

Product → MANUFACTURES → Supplier does not.

If we used a standard nx.Graph() (Undirected), the system would treat those relationships as two-way streets, which would confuse our risk analysis later.

💻 Step 2: Initialize the Graph

Now, let's write the code for graph_ops.py. We need to:

Import networkx.

Import our node classes from schema.

Initialize the graph.

The code for graph_ops.py that includes a class (let's call it SupplyChainGraph) and an __init__ method that sets up the self.graph using nx.DiGraph()
"""

import networkx as nx
from schema import Supplier, Product, Location

class SupplyChainGraph:
    def __init__(self):
        """
        Initialize the graph using nx.DiGraph().
        """
        self.graph = nx.DiGraph()

    def add_node(self, node: Supplier | Product | Location):
        """
        Add a node to the graph.
        """
        self.graph.add_node(node)

    def add_edge(self, source: Supplier | Product | Location, target: Supplier | Product | Location, edge_type: str):
        """
        Add an edge to the graph.
        """
        self.graph.add_edge(source, target, edge_type=edge_type)

    def get_graph(self):
        """
        Get the graph.
        """
        return self.graph

    def get_nodes(self):
        """
        Get the nodes of the graph.
        """
        return list(self.graph.nodes)