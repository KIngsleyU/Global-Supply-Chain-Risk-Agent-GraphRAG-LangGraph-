# NetworkX Logic (The World)

"""
my plan implementation for this file

You are absolutely right to use nx.DiGraph() (Directed Graph). In our supply chain, relationships are asymmetric:

Supplier → MANUFACTURES → Product makes sense.

Product → MANUFACTURES → Supplier does not.

If we used a standard nx.Graph() (Undirected), the system would treat those relationships as two-way streets, which would confuse our risk analysis later.

💻 Step 2: Initialize the Graph

Now, let's write the code for graph_ops.py. We need to:

Import networkx.

Import our node classes from schema.

Initialize the graph.

The code for graph_ops.py that includes a class (let's call it SupplyChainGraph) and an __init__ method that sets up the self.graph using nx.DiGraph()
"""

"""
next steps plan implementation for this file

Generating the Synthetic World

Now, let's write the generate_data function inside graph_ops.py.

We need to use the Faker library to create realistic dummy data. We will need to generate three layers of data in this specific order:

Locations: Create 5-10 random locations (e.g., "Port of Shanghai", "Hamburg Warehouse").

Suppliers: Create ~20 suppliers. For each one, pick a random Location from the list you just made and add the LOCATED_AT edge.

Products: Create ~50 products. For each one, pick a random Supplier and add the MANUFACTURES edge.

The Challenge: Can you write the generate_data(graph_instance) function?

Hint: You'll need import random and from faker import Faker.

Hint: Remember to instantiate the classes (Location(...), Supplier(...)) before adding them to the graph.

Give it a try! Don't worry about syntax errors; we'll fix them together.
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
                product = Product(
                    name=self.faker.product_name(), 
                    sku=self.faker.unique.random_int(min=1000000000, max=9999999999), 
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