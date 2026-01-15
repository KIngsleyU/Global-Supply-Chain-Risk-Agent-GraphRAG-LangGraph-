# The Entry Point

from graph_ops import SupplyChainGraph
from vector_ops import ProductVectorStore

graph = SupplyChainGraph()
vector_store = ProductVectorStore()

graph.generate_data()
vector_store.add_products(graph.get_products())

print(vector_store.get_products("medical equipment"))