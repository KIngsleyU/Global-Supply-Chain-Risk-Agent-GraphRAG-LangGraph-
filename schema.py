# Our Data Models (Nodes)

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

@dataclass
class Location:
    name: str
    country: str  # Added the missing 'str' type hint