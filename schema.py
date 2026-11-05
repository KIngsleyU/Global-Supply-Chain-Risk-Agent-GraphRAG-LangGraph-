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

from dataclasses import dataclass, field

@dataclass(frozen=True)
class Supplier:
    name: str
    risk_score: float = field(compare=False)
    revenue: float = field(compare=False)

@dataclass(frozen=True)
class Product:
    name: str = field(compare=False)
    sku: str
    price: float = field(compare=False)

@dataclass(frozen=True)
class Location:
    name: str
    country: str