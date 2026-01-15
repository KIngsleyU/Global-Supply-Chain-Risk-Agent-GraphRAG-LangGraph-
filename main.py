# The Entry Point

from dataclasses import dataclass

@dataclass
class Supplier:
    name: str
    risk_score: float
    revenue: float

@dataclass
class Product:
    name: str
    sku: str
    price: float

@dataclass
class Location:
    name: str
    country: str
    
