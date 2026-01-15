# Our Data Models (Nodes)

"""
Schema Module for Supply Chain Risk Analysis

This module defines the core data models (nodes) for the supply chain knowledge graph.
All classes are implemented as Python dataclasses with hash methods to enable their use
as nodes in NetworkX graphs.

The module provides three main entity types:
- Supplier: Represents manufacturing companies with risk attributes
- Product: Represents manufactured goods with identifiers and pricing
- Location: Represents geographical regions where suppliers operate

All classes implement __hash__ methods using unique identifiers to ensure they can be
used as dictionary keys in NetworkX graph operations:
- Supplier: Uses name as the unique identifier
- Product: Uses SKU (Stock Keeping Unit) as the unique identifier
- Location: Uses (name, country) tuple for uniqueness

Example:
    >>> from schema import Supplier, Product, Location
    >>> supplier = Supplier(name="Acme Corp", risk_score=0.3, revenue=1000000)
    >>> product = Product(name="Widget A", sku="WID-A001", price=99.99)
    >>> location = Location(name="Port of Los Angeles", country="United States")
"""

from dataclasses import dataclass

@dataclass
class Supplier:
    name: str
    risk_score: float
    revenue: float
    # We add a unique ID helper for the graph
    def __hash__(self):
        return hash(self.name)

@dataclass
class Product:
    name: str
    sku: str
    price: float
    # We add a unique ID helper for the graph (SKU should be unique)
    def __hash__(self):
        return hash(self.sku)

@dataclass
class Location:
    name: str
    country: str  # Added the missing 'str' type hint
    # We add a unique ID helper for the graph
    def __hash__(self):
        return hash((self.name, self.country))