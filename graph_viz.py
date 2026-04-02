"""
Graph Visualization Utilities for Supply Chain Risk Analysis.

This module centralizes graph visualization logic so it can be reused by:
- Streamlit UI (app.py)
- LangGraph tools (agent.py)
"""

import graphviz

from graph_ops import SupplyChainGraph
import streamlit as st



def build_supply_chain_graphviz(graph_db: SupplyChainGraph, risk_threshold: float = 0.6) -> graphviz.Digraph:
    """
    Build a Graphviz directed graph for the generated supply chain.

    Args:
        graph_db: SupplyChainGraph instance with generated locations/suppliers/products.
        risk_threshold: Suppliers above this risk are colored red.

    Returns:
        graphviz.Digraph: Renderable graph object.
    """
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")

    # A. Draw locations (blue boxes)
    for loc in graph_db.locations:
        dot.node(loc.name, label=loc.name, shape="box", style="filled", fillcolor="lightblue")

    # B. Draw suppliers (red if risk > threshold, otherwise green)
    for sup in graph_db.suppliers:
        if sup.risk_score > risk_threshold:
            fillcolor = "red"
            label = f"{sup.name}\\n(Risk: {sup.risk_score})"
        else:
            fillcolor = "lightgreen"
            label = sup.name

        dot.node(sup.name, label=label, style="filled", fillcolor=fillcolor)

        # Supplier -> Location edges
        for neighbor in graph_db.graph.successors(sup):
            if hasattr(neighbor, "country"):
                dot.edge(sup.name, neighbor.name, label="LOCATED_AT")

    # C. Draw products (yellow ovals)
    for prod in graph_db.products:
        dot.node(prod.name, label=f"{prod.name}\\n${prod.price}", style="filled", fillcolor="lightyellow")

        # Supplier -> Product edges (rendered as MAKES for readability)
        for supplier in graph_db.graph.predecessors(prod):
            dot.edge(supplier.name, prod.name, label="MAKES")

    return dot

if __name__ == "__main__":
    # display the graph
    graph_db = SupplyChainGraph()   
    graph_db.generate_data()
    dot = build_supply_chain_graphviz(graph_db, risk_threshold=0.6)
    print(dot.source)
    dot.render('supply_chain_graph.png', view=True)
    st.graphviz_chart(dot, use_container_width=True)