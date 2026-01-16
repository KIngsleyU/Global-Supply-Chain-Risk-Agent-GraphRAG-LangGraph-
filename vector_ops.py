# ChromaDB Logic (The Brain)

"""
Vector Operations Module for Semantic Search

This module provides vector database functionality for semantic search over supply chain entities
using ChromaDB. It enables finding products, suppliers, and locations based on semantic similarity
rather than exact keyword matches, which is crucial for risk analysis when querying with natural
language descriptions, misspellings, or partial matches.

The module provides three vector store classes:
- ProductVectorStore: Semantic search for products
- SupplierVectorStore: Semantic search for suppliers
- LocationVectorStore: Semantic search for locations

Design Decisions:
- Uses entity names for embedding: Names contain semantic information (e.g., 
  "Surgical Mask", "Port of Shanghai", "Acme Corp") that allows the embedding model to map
  concepts even without exact word matches.
- Unique IDs stored in metadata: Identifiers (SKU, supplier name, location name+country) are
  stored in metadata rather than embedded since they have no semantic meaning.
- In-memory storage: Uses EphemeralClient for in-memory operation, consistent with the
  project's NetworkX approach of running entirely in RAM.

Example:
    >>> from vector_ops import ProductVectorStore, SupplierVectorStore, LocationVectorStore
    >>> from schema import Product, Supplier, Location
    >>> 
    >>> product_store = ProductVectorStore()
    >>> products = [Product(name="Surgical Mask", sku="SM001", price=5.99)]
    >>> product_store.add_products(products)
    >>> results = product_store.get_products("medical equipment", k=5)
"""

import chromadb
from schema import Product, Supplier, Location

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


class SupplierVectorStore:
    def __init__(self):
        """Initialize a vector store for suppliers."""
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("suppliers")

    def add_suppliers(self, suppliers: list[Supplier]):
        """
        Add suppliers to the collection.
        suppliers: list[Supplier]
        return: bool
        
        example:
        suppliers = [Supplier(name="Acme Corp", risk_score=0.3, revenue=1000000)]
        vector_store.add_suppliers(suppliers)
        return True
        """
        ids = []
        documents = []
        metadatas = []
        for supplier in suppliers:
            # Use supplier name as unique ID
            ids.append(supplier.name)
            documents.append(supplier.name)  # Embed the name for semantic search
            metadatas.append({
                "name": supplier.name,
                "risk_score": supplier.risk_score,
                "revenue": supplier.revenue
            })
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Indexed {len(ids)} suppliers in ChromaDB.")
            return True
        else:
            print("No suppliers to index.")
            return False
    
    def get_suppliers(self, query: str, k: int = 5):
        """
        Search for suppliers using semantic similarity search.
        Useful for finding suppliers by name when there are misspellings or partial matches.
        
        query: str - Search query text (e.g., "Acme", "Mexico City supplier")
        k: int - Number of top results to return (default: 5)
        return: list[Supplier] - List of Supplier objects matching the query
        
        example:
            suppliers = vector_store.get_suppliers("Acme Corp", k=3)
            # Returns top 3 suppliers semantically similar to "Acme Corp"
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        found_suppliers = []
        if results and results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            ids = results['ids'][0]
            
            for i in range(len(docs)):
                found_suppliers.append(Supplier(
                    name=docs[i],
                    risk_score=metas[i]['risk_score'],
                    revenue=metas[i]['revenue']
                ))
        return found_suppliers
    
    def get_all_suppliers(self, names: list[str]):
        """
        Retrieve suppliers from the collection by their names.
        
        names: list[str] - List of supplier name strings to retrieve
        return: list[Supplier] - List of Supplier objects
        """
        results = self.collection.get(ids=names)
        found_suppliers = []
        
        if results and results.get('documents') and len(results['documents']) > 0:
            docs = results['documents']
            metas = results['metadatas']
            result_ids = results['ids']
            
            for i in range(len(result_ids)):
                found_suppliers.append(Supplier(
                    name=docs[i],
                    risk_score=metas[i]['risk_score'],
                    revenue=metas[i]['revenue']
                ))
        
        return found_suppliers


class LocationVectorStore:
    def __init__(self):
        """Initialize a vector store for locations."""
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("locations")

    def add_locations(self, locations: list[Location]):
        """
        Add locations to the collection.
        locations: list[Location]
        return: bool
        
        example:
        locations = [Location(name="Port of Shanghai", country="China")]
        vector_store.add_locations(locations)
        return True
        """
        ids = []
        documents = []
        metadatas = []
        for location in set(locations):
            # Use name + country as unique ID (matches Location.__hash__)
            unique_id = f"{location.name}::{location.country}"
            ids.append(unique_id)
            # Embed both name and country for better semantic search
            documents.append(f"{location.name} {location.country}")
            metadatas.append({
                "name": location.name,
                "country": location.country
            })
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Indexed {len(ids)} locations in ChromaDB.")
            return True
        else:
            print("No locations to index.")
            return False
    
    def get_locations(self, query: str, k: int = 5):
        """
        Search for locations using semantic similarity search.
        Useful for finding locations by name when there are misspellings, partial matches,
        or variations like "Port of mexico city" vs "Mexico City Manufacturing Facility".
        
        query: str - Search query text (e.g., "Port of Shanghai", "Mexico City", "Shanghai")
        k: int - Number of top results to return (default: 5)
        return: list[Location] - List of Location objects matching the query
        
        example:
            locations = vector_store.get_locations("Port of mexico city", k=3)
            # Returns top 3 locations semantically similar to the query
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        
        found_locations = []
        if results and results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            ids = results['ids'][0]
            
            for i in range(len(docs)):
                found_locations.append(Location(
                    name=metas[i]['name'],
                    country=metas[i]['country']
                ))
        return found_locations
    
    def get_all_locations(self, ids: list[str]):
        """
        Retrieve locations from the collection by their unique IDs (name::country format).
        
        ids: list[str] - List of location ID strings in "name::country" format
        return: list[Location] - List of Location objects
        """
        results = self.collection.get(ids=ids)
        found_locations = []
        
        if results and results.get('documents') and len(results['documents']) > 0:
            metas = results['metadatas']
            
            for i in range(len(results['ids'])):
                found_locations.append(Location(
                    name=metas[i]['name'],
                    country=metas[i]['country']
                ))
        
        return found_locations