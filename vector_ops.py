# ChromaDB Logic (The Brain)

"""
Vector Operations Module for Semantic Product Search

This module provides vector database functionality for semantic search over product catalogs
using ChromaDB. It enables finding products based on semantic similarity rather than exact
keyword matches, which is crucial for risk analysis when querying with natural language
descriptions of products or risk categories.

The ProductVectorStore class enables:
- Indexing products by their names (semantically meaningful text)
- Semantic search to find products matching natural language queries
- Retrieval of products by SKU for direct lookups
- Management of product metadata (SKU, price) alongside embeddings

Design Decisions:
- Uses product names for embedding: Product names contain semantic information (e.g., 
  "Surgical Mask", "Hydraulic Pump") that allows the embedding model to map concepts like
  "medical supplies" to relevant products even without exact word matches.
- SKUs stored in metadata: SKUs are numeric identifiers with no semantic meaning, so they
  are stored in metadata rather than embedded.
- In-memory storage: Uses EphemeralClient for in-memory operation, consistent with the
  project's NetworkX approach of running entirely in RAM.

Example:
    >>> from vector_ops import ProductVectorStore
    >>> from schema import Product
    >>> 
    >>> store = ProductVectorStore()
    >>> products = [Product(name="Surgical Mask", sku="SM001", price=5.99)]
    >>> store.add_products(products)
    >>> results = store.get_products("medical equipment", k=5)
"""

import chromadb
from schema import Product

class ProductVectorStore:
    def __init__(self):
        # Use EphemeralClient for in-memory storage (consistent with NetworkX approach)
        # This matches the project philosophy of running entirely in RAM
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("products")

    def add_products(self, products: list[Product]):
        """
        Add products to the collection.
        products: list[Product]
        return: bool
        
        example:
        products = [Product(name="Product 1", sku="1234567890", price=100), Product(name="Product 2", sku="1234567891", price=200)]
        vector_store.add_products(products)
        return True
        """
        ids = []
        documents = []
        metadatas = []
        for product in products:
            ids.append(product.sku)
            documents.append(product.name)
            metadatas.append({
                "name": product.name, 
                "sku": product.sku, 
                "price": product.price})
        if ids:
            self.collection.add(
                ids=ids, 
                documents=documents, 
                metadatas=metadatas)
            print(f"Indexed {len(ids)} products in ChromaDB.")
            return True
        else:
            print("No products to index.")
            return False
    
    def get_products(self, query: str, k: int = 5):
        """
        Search for products using semantic similarity search.
        Uses vector embeddings to find products whose names are semantically similar to the query.
        
        query: str - Search query text (e.g., "hazardous chemicals", "medical supplies")
        k: int - Number of top results to return (default: 5)
        return: list[Product] - List of Product objects matching the query
        
        example:
            products = vector_store.get_products("medical equipment", k=3)
            # Returns top 3 products semantically similar to "medical equipment"
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        found_products = []
        # Check if we actually found anything
        if results and results['documents']:
            # Access index [0] because we sent 1 query string
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            ids = results['ids'][0]
            
            for i in range(len(docs)):
                found_products.append(Product(
                    name=docs[i],
                    sku=ids[i],
                    price=metas[i]['price']
                ))
        return found_products
    
    def get_all_products(self, ids: list[str]):
        """
        Retrieve products from the collection by their SKU IDs.
        Note: collection.get() returns flat lists (not nested like query()).
        
        ids: list[str] - List of SKU strings to retrieve
        return: list[Product] - List of Product objects
        
        example:
            products = vector_store.get_all_products(["1234567890", "1234567891"])
            # Returns [Product(name="Product 1", sku="1234567890", price=100), 
            #          Product(name="Product 2", sku="1234567891", price=200)]
        """
        results = self.collection.get(ids=ids)
        found_products = []
        
        # collection.get() returns flat lists, not nested like query()
        if results and results.get('documents') and len(results['documents']) > 0:
            docs = results['documents']
            metas = results['metadatas']
            result_ids = results['ids']
            
            # All lists should be the same length
            for i in range(len(result_ids)):
                found_products.append(Product(
                    name=docs[i],
                    sku=result_ids[i],
                    price=metas[i]['price']
                ))
        
        return found_products
    
    def delete_all_products(self, ids: list[str]):
        """
        Delete all products from the collection.
        ids: list[str]
        return: bool
        
        example:
        vector_store.delete_all_products(["1234567890", "1234567891"])  # type: ignore
        return True
        """
        self.collection.delete(ids=ids)
        return True