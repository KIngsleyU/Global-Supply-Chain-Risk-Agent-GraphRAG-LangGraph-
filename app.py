"""
Streamlit Web Application for Supply Chain Risk Visualization

This module provides an interactive web interface for visualizing the supply chain knowledge graph
using Streamlit and Graphviz. It enables users to explore supply chain relationships, identify
high-risk suppliers, and understand the connections between locations, suppliers, and products.

Key Features:
- Interactive web-based visualization of the supply chain graph
- Risk threshold slider to highlight suppliers with risk scores above a configurable threshold
- Color-coded visualization:
  * Blue boxes: Locations
  * Red boxes: High-risk suppliers (risk_score > threshold)
  * Green boxes: Low-risk suppliers (risk_score <= threshold)
  * Yellow ovals: Products with pricing information
- Directed graph visualization showing relationships:
  * Supplier → LOCATED_AT → Location
  * Supplier → MANUFACTURES → Product

Architecture:
- Uses Streamlit for web interface and user interaction
- Uses Graphviz via streamlit's graphviz_chart for graph rendering
- Integrates with SupplyChainGraph from graph_ops module for data access
- Caches graph data using @st.cache_resource for performance

Usage:
    Run the Streamlit application:
    $ streamlit run app.py
    
    The application will:
    1. Generate or load supply chain data using SupplyChainGraph
    2. Display an interactive graph visualization
    3. Allow users to adjust risk threshold via sidebar slider
    4. Update the visualization to highlight suppliers based on risk threshold

Dependencies:
    - streamlit: Web application framework
    - graphviz: Graph visualization library
    - graph_ops: Supply chain graph operations module
"""

import streamlit as st
import graphviz
from graph_ops import SupplyChainGraph

# 1. Setup the Page
st.set_page_config(page_title="Supply Chain Risk Guardian", layout="wide")
st.title("🌍 Global Supply Chain Risk Map")

# 2. Initialize the World (using your existing logic!)
@st.cache_resource
def load_graph():
    graph_db = SupplyChainGraph()
    graph_db.generate_data()
    return graph_db

graph_db = load_graph()

# 3. Sidebar Controls
risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.6)

# 4. Build the Visual Graph
def draw_supply_chain(graph, threshold):
    # Create a directed graph
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR') # Left to Right layout

    # A. Draw Locations (Blue Boxes)
    for loc in graph.locations:
        dot.node(loc.name, label=loc.name, shape='box', style='filled', fillcolor='lightblue')

    # B. Draw Suppliers (Red vs Green Logic)
    for sup in graph.suppliers:
        # --- YOUR LOGIC HERE ---
        if sup.risk_score > threshold:
            color = 'red'
            label = f"{sup.name}\n(Risk: {sup.risk_score})"
        else:
            color = 'lightgreen'
            label = sup.name
        
        dot.node(sup.name, label=label, style='filled', fillcolor=color)
        
        # Link Supplier -> Location
        # In our schema we stored the edge, but for viz we can just use the logic we know
        # We need to find which location this supplier is at.
        # Since we didn't store 'location_name' on the supplier object explicitly in the final schema, 
        # we can look at the graph edges!
        
        # Find the edge: Supplier -> Location
        for neighbor in graph.graph.successors(sup):
            # Check if neighbor is a Location object
            if hasattr(neighbor, 'country'): 
                dot.edge(sup.name, neighbor.name, label="LOCATED_AT")

    # C. Draw Products (Yellow Ovals)
    for prod in graph.products:
        dot.node(prod.name, label=f"{prod.name}\n${prod.price}", style='filled', fillcolor='lightyellow')
        
        # Find the edge: Supplier -> Product
        # In our schema: Supplier -> MANUFACTURES -> Product
        # So we look for predecessors of the product
        for supplier in graph.graph.predecessors(prod):
            dot.edge(supplier.name, prod.name, label="MAKES")

    return dot

# 5. Render
st.graphviz_chart(draw_supply_chain(graph_db, risk_threshold), use_container_width=True)

st.markdown(f"### 🔍 Analysis")
st.write(f"Showing suppliers with Risk Score > {risk_threshold} in **RED**.")