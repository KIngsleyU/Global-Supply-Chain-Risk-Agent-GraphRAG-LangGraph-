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
import os
from graph_ops import SupplyChainGraph
from graph_viz import build_supply_chain_graphviz

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

# 5. Render
# If agent.py launched this app, render the serialized DOT snapshot to avoid reinitializing
# a separate graph instance for visualization.
dot_path = os.getenv("SC_GRAPH_DOT_PATH")
if dot_path and os.path.exists(dot_path):
    with open(dot_path, "r", encoding="utf-8") as f:
        dot_source = f.read()
    st.graphviz_chart(dot_source, use_container_width=True)
    st.caption(f"Rendering graph snapshot from agent.py ({dot_path})")
else:
    dot = build_supply_chain_graphviz(graph_db, risk_threshold=risk_threshold)
    st.graphviz_chart(dot, use_container_width=True)

st.markdown(f"### 🔍 Analysis")
st.write(f"Showing suppliers with Risk Score > {risk_threshold} in **RED**.")