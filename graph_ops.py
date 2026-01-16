# NetworkX Logic (The World)
"""
Graph Operations Module for Supply Chain Risk Analysis

This module provides the core graph infrastructure for modeling and analyzing global supply chains
using NetworkX. It implements a knowledge graph representation where suppliers, products, and
locations are nodes connected by directed edges representing business relationships.

The SupplyChainGraph class enables:
- Creation and management of a directed graph (DiGraph) to model asymmetric supply chain relationships
- Synthetic data generation for testing and demonstration purposes
- Graph traversal and query operations for risk analysis

Key Features:
- Directed graph structure ensures proper modeling of asymmetric relationships:
  * Supplier → MANUFACTURES → Product (one-way relationship)
  * Supplier → LOCATED_AT → Location (one-way relationship)
- Comprehensive data generation pipeline that creates realistic supply chain networks:
  * 20-30 diverse locations (ports, warehouses, manufacturing facilities)
  * 1-3 suppliers per location with risk scores and revenue data
  * 1-3 products per supplier with SKUs and pricing
- State management with helper methods to access generated entities

Example:
    >>> from graph_ops import SupplyChainGraph
    >>> graph = SupplyChainGraph()
    >>> graph.generate_data()
    Generated 25 locations, 48 suppliers, 92 products
    >>> nx_graph = graph.get_graph()
    >>> locations = graph.get_locations()
"""

import random
from faker import Faker

import networkx as nx
from schema import Supplier, Product, Location


class SupplyChainGraph:
    def __init__(self):
        """
        Initialize the graph using nx.DiGraph().
        """
        self.graph = nx.DiGraph()
        self.locations = []
        self.suppliers = []
        self.products = []
        self.faker = Faker()

    def add_node(self, node: Supplier | Product | Location):
        """
        Add a node to the graph.
        """
        self.graph.add_node(node)

    def add_edge(self, source: Supplier | Product | Location, target: Supplier | Product | Location, edge_type: str):
        """
        Add an edge to the graph.
        """
        self.graph.add_edge(source, target, edge_type=edge_type)

    def get_graph(self):
        """
        Get the graph.
        """
        return self.graph

    def get_nodes(self):
        """
        Get the nodes of the graph.
        return: list[Supplier | Product | Location]
        
        example:
        nodes = get_nodes()
        return nodes
        """
        return list(self.graph.nodes)
    
    def get_node_by_name(self, name: str):
        """
        Get a node by name.
        name: str
        return: Supplier | Product | Location
        
        example:
        node = get_node_by_name("Port of Shanghai")
        return node
        """
        return next((node for node in self.graph.nodes if node.name == name), None)
    
    def generate_locations(self):
        """
        Generate 20-30 diverse, realistic supply chain locations for the graph.
        Includes ports, warehouses, manufacturing facilities, and distribution centers.
        
        Returns:
            list[Location]: List of generated Location objects
        
        Example:
            locations = self.generate_locations()
            # Returns list of Location objects like:
            # [Location(name="Port of Shanghai", country="China"), 
            #  Location(name="Hamburg Warehouse", country="Germany"), ...]
        """
        
        locations = []
        
        # Define realistic supply chain location templates with cities
        location_templates = [
            # Major ports - format: "Port of {city}" or "{city} Port"
            (["Port of {}", "{} Port"], ["Shanghai", "Los Angeles", "Rotterdam", "Singapore", 
                                          "Hamburg", "Antwerp", "Hong Kong", "Busan", "Dubai", 
                                          "Long Beach", "Tianjin", "Ningbo-Zhoushan", "Qingdao"]),
            # Warehouses - format: "{city} Warehouse", "{city} Distribution Center", etc.
            (["{} Warehouse", "{} Distribution Center", "{} Logistics Hub"], 
             ["Hamburg", "Frankfurt", "Chicago", "Memphis", "Atlanta", "Tokyo", 
              "Seoul", "Mumbai", "São Paulo", "Amsterdam", "London", "Paris"]),
            # Manufacturing facilities - format: "{city} Manufacturing Facility", etc.
            (["{} Manufacturing Facility", "{} Production Center", "{} Factory"], 
             ["Shenzhen", "Guangzhou", "Bangalore", "Ho Chi Minh City", "Manila", 
              "Jakarta", "Bangkok", "Istanbul", "Mexico City", "San Diego"]),
        ]
        
        # Generate 20-30 locations
        num_locations = random.randint(20, 30)
        
        for _ in range(num_locations):
            # Randomly select a location type (port, warehouse, or manufacturing)
            location_type = random.choice(location_templates)
            templates, city_names = location_type
            
            # Randomly select a template and city
            template = random.choice(templates)
            city = random.choice(city_names)
            
            # Format the location name (all templates now use {} placeholder)
            location_name = template.format(city)
            
            # Get country from city (simplified mapping for realism)
            country_mapping = {
                "Shanghai": "China", "Tianjin": "China", "Ningbo-Zhoushan": "China", 
                "Qingdao": "China", "Hong Kong": "China", "Shenzhen": "China", 
                "Guangzhou": "China",
                "Los Angeles": "United States", "Long Beach": "United States", 
                "Chicago": "United States", "Memphis": "United States", 
                "Atlanta": "United States", "San Diego": "United States",
                "Rotterdam": "Netherlands", "Amsterdam": "Netherlands",
                "Hamburg": "Germany", "Frankfurt": "Germany",
                "Singapore": "Singapore",
                "Busan": "South Korea", "Seoul": "South Korea",
                "Dubai": "United Arab Emirates",
                "Antwerp": "Belgium",
                "Tokyo": "Japan",
                "Mumbai": "India", "Bangalore": "India",
                "São Paulo": "Brazil",
                "London": "United Kingdom",
                "Paris": "France",
                "Ho Chi Minh City": "Vietnam",
                "Manila": "Philippines",
                "Jakarta": "Indonesia",
                "Bangkok": "Thailand",
                "Istanbul": "Turkey",
                "Mexico City": "Mexico"
            }
            
            country = country_mapping.get(city, self.faker.country())
            
            location = Location(name=location_name, country=country)
            self.add_node(location)
            locations.append(location)
        
        return locations
    
    def generate_suppliers(self, locations: list[Location]):
        """
        Generate the suppliers for the graph.
        locations: list[Location]
        return: list[Supplier]
        
        example:
        locations = [Location(name="Port of Shanghai", country="China"), Location(name="Port of Hamburg", country="Germany")]
        suppliers = generate_suppliers(locations)
        return suppliers
        """
        for location in locations:
            # generate 1-3 suppliers for each location
            for _ in range(random.randint(1, 3)):
                supplier = Supplier(
                    name=self.faker.company(), 
                    risk_score=round(random.uniform(0, 1), 2), 
                    revenue=round(random.uniform(100000, 10000000), 2))
                self.suppliers.append(supplier)
                self.add_node(supplier) # type: Supplier ignore
                self.add_edge(supplier, location, "LOCATED_AT") # type: Supplier ignore
        return self.suppliers
    
    def generate_products(self, suppliers: list[Supplier]):
        """
        Generate the products for the graph.
        suppliers: list[Supplier]
        return: list[Product]

        example:
        suppliers = [Supplier(name="Supplier 1", risk_score=0.5, revenue=1000000, location="Port of Shanghai"), Supplier(name="Supplier 2", risk_score=0.3, revenue=2000000, location="Port of Hamburg")]
        products = generate_products(suppliers)
        return products
        """
        for supplier in suppliers:
            # generate 1-3 products for each supplier
            for _ in range(random.randint(1, 3)):
                # Use Faker's unique to ensure SKUs are unique
                sku = str(self.faker.unique.random_int(min=1000, max=99999))
                
                product = Product(
                    name=self.faker.bs().title(), # 'bs' gives catchy product-like names
                    sku=sku,
                    price=round(random.uniform(100, 1000), 2))
                self.add_node(product) # type: Product ignore
                self.add_edge(supplier, product, "MANUFACTURES") # type: Product ignore
                self.products.append(product)
        return self.products
    
    def generate_data(self):
        """
        Generate the data for the graph.
        return: list[Supplier], list[Product], list[Location]
        
        example:
        graph = generate_data()
        return graph
        """
        self.locations = self.generate_locations()
        self.suppliers = self.generate_suppliers(self.locations)
        self.products = self.generate_products(self.suppliers)
        print(f"Generated {len(self.locations)} locations, {len(self.suppliers)} suppliers, {len(self.products)} products")
        # return self.locations, self.suppliers, self.products
        return self.graph
    
    def get_locations(self):
        """
        Get the locations of the graph.
        return: list[Location]
        """
        return self.locations
    
    def get_suppliers(self):
        """
        Get the suppliers of the graph.
        return: list[Supplier]
        """
        return self.suppliers
    
    def get_products(self):
        """
        Get the products of the graph.
        return: list[Product]
        """
        return self.products