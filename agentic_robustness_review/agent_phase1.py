"""
Phase 1 + Phase 2 + Phase 3 Upgraded LangGraph Agent:
- Phase 1: Multi-Hop Subgraph Traversal & Deterministic Aggregation
- Phase 2: Structured `AgentState` Working Memory & Duplicate Tool-Call Guard
- Phase 3: Corrective RAG (CRAG) / Deterministic Completeness & Numerical Grounding Verifier Node

Builds directly on top of the existing root modules (`schema.py`, `graph_ops.py`,
`vector_ops.py`, `graph_viz.py`) while keeping the original `agent.py` and `main.py`
untouched.

Key Phase 1 Improvements over Baseline `agent.py`:
1. `analyze_location_blast_radius(location_or_country_query, risk_threshold=0.6)`:
   - Resolves exact, partial, or misspelled city/country queries (via hybrid substring +
     ChromaDB vector search).
   - Performs a complete 2-hop directed NetworkX traversal:
     `Location <-[:LOCATED_AT]- Supplier -[:MANUFACTURES]-> Product` in <1 ms.
   - Deterministically calculates exact `total_revenue_exposed`, `risk_weighted_exposure`,
     and high-risk supplier counts in Python—eliminating LLM mental-math hallucinations
     and reducing 15–20 sequential tool calls down to 1.
2. `audit_products_and_upstream_suppliers(product_query, risk_threshold=0.6, k=25)`:
   - Removes the hardcoded `k=5` retrieval ceiling and adds catalog-aware keyword matching
     so no relevant products are silently truncated.
   - Automatically traverses 2 hops upstream (`Product <- Supplier -> Location`) and flags
     all suppliers with `risk_score > risk_threshold` sorted by severity.
3. Accurate Tool Call Accounting:
   - Counts total tool invocations (rather than just AI turns) to prevent runaway loops.

Key Phase 2 Improvements (Structured `AgentState` Working Memory):
4. Replaces flat `messages`-only state with a structured `AgentState` containing:
   - `visited_nodes`: Canonical set of executed tool signatures & resolved entity names
     so duplicate tool calls across multi-model OpenRouter turns are intercepted and
     served from state without re-running tools.
   - `discovered_locations`, `discovered_suppliers`, `discovered_products`: Canonical
     entity dictionaries deduplicated by name/SKU across turns.
   - `deterministic_metrics`: Latest Python-computed financial and count metrics stored
     directly in graph state and injected into `chatbot` as a compact working-memory
     digest so any routed LLM has instant access without re-parsing raw JSON `ToolMessage`s.

Key Phase 3 Improvements (CRAG / Self-Reflection Verifier Node):
5. `verify_and_reflect(state)` & Conditional Routing (`route_after_chatbot`, `route_after_verify`):
   - Instead of `chatbot` transitioning straight to `__end__` when it finishes calling tools,
     its candidate final report is routed to `verify_and_reflect`.
   - Runs 2 fast, deterministic Python checks against `AgentState`:
     a) Entity Completeness Check: Verifies that every discovered Location, Supplier, and
        Product (by name or SKU) is present in the final answer.
     b) Numerical Grounding Check: Verifies that the exact Python-computed total revenue
        exposure (`total_supplier_revenue_exposed_formatted`) is quoted in the report.
   - If any entity is omitted or the total revenue is missing/misquoted (and
     `reflection_count < 2`), `verify_and_reflect` sets `reflection_feedback`, appends a
     targeted correction instruction, increments `reflection_count`, and routes back to
     `chatbot` to self-correct before finishing.
"""

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import networkx as nx

from graph_ops import SupplyChainGraph
from graph_viz import build_supply_chain_graphviz
from schema import Location, Product, Supplier
from vector_ops import LocationVectorStore, ProductVectorStore, SupplierVectorStore

load_dotenv(REPO_ROOT / ".env")

# Initialize Knowledge Graph and ChromaDB Vector Stores using existing root classes
graph_db = SupplyChainGraph()
graph_db.generate_data()

product_vector_db = ProductVectorStore()
supplier_vector_db = SupplierVectorStore()
location_vector_db = LocationVectorStore()

product_vector_db.add_products(graph_db.products)
supplier_vector_db.add_suppliers(graph_db.suppliers)
location_vector_db.add_locations(graph_db.locations)

# Known product catalog lines by category (mirrors graph_ops.py for deterministic category matching)
CATEGORY_KEYWORDS = {
    "electronics": [
        "printed circuit board", "power supply unit", "semiconductor sensor",
        "industrial router", "battery management module", "servo controller",
    ],
    "automotive": [
        "brake disc assembly", "fuel injection pump", "wiring harness",
        "transmission gear set", "engine control unit", "steering column module",
    ],
    "pharmaceutical": [
        "active pharmaceutical ingredient", "sterile syringe kit", "diagnostic reagent pack",
        "medical-grade nitrile gloves", "surgical mask carton", "cold chain vaccine carrier",
    ],
    "medical": [
        "active pharmaceutical ingredient", "sterile syringe kit", "diagnostic reagent pack",
        "medical-grade nitrile gloves", "surgical mask carton", "cold chain vaccine carrier",
    ],
    "industrial": [
        "hydraulic pump", "ball bearing set", "stainless steel fastener kit",
        "conveyor belt roll", "cnc spindle motor", "pressure relief valve",
    ],
    "packaging": [
        "corrugated shipping carton", "pallet wrap film", "insulated shipping box",
        "tamper-evident seal roll", "label stock roll", "reusable plastic crate",
    ],
    "energy": [
        "solar inverter unit", "lithium-ion cell pack", "high-voltage cable harness",
        "grid monitoring meter", "wind turbine bearing", "ev charging connector",
    ],
}


LOCATION_STOPWORDS = {
    "port", "ports", "warehouse", "warehouses", "facility", "facilities",
    "center", "centers", "hub", "hubs", "factory", "factories",
    "distribution", "manufacturing", "production", "logistics",
    "location", "locations", "all", "the", "and", "for", "near", "in", "at", "of",
}


def _resolve_locations_hybrid(query: str) -> tuple[list[Location], str]:
    """
    Resolve a location, city, or country query (including typos like 'Chna', 'shanghia')
    into matching Location nodes from `graph_db` without false-positive substring matches
    (e.g., preventing 'San' in 'San Diego' from matching 'Busan').

    ===========================================================================
    WALKTHROUGH WITH 2 RUNNING TEST CASES
    ===========================================================================
    Assume `graph_db.locations` contains these 3 nodes:
      L1 = Location(name="San Diego Factory", country="United States")
      L2 = Location(name="Shanghai Port",     country="China")
      L3 = Location(name="Port of Busan",     country="South Korea")

    • Test Case A (Clean phrase):  query = "Port of San Diego"
      -> Step 0 strips "port", "of" -> clean_geo_query = "san diego"
      -> Step 1 matches whole phrase "\\bsan diego\\b" in L1.name ("san diego factory")
      -> Returns: ([L1], "direct_match")

    • Test Case B (Misspelled):    query = "ports in Chna"
      -> Step 0 strips "ports", "in" -> query_words = ["chna"], clean_geo_query = "chna"
      -> Step 1 (Exact phrase): "\\bchna\\b" not in L1, L2, L3 -> direct_matches = []
      -> Step 2 (Exact word):   "chna" not in loc_words of L1, L2, L3 -> token_matches = []
      -> Step 3 (Fuzzy ratio):  SequenceMatcher("chna", "china").ratio() = 0.89 (>= 0.72)
         -> fuzzy_matched_countries = {"China"}
      -> Returns: ([L2], "fuzzy_country_match(China)")
    ===========================================================================
    """
    # Normalize input query to lowercase and strip outer whitespace
    # Case A: "Port of San Diego" -> q_lower = "port of san diego"
    # Case B: "ports in Chna"     -> q_lower = "ports in chna"
    q_lower = query.lower().strip()

    # Fetch all Location dataclass instances currently in the graph
    # Output: [Location("San Diego Factory", "United States"), Location("Shanghai Port", "China"), Location("Port of Busan", "South Korea")]
    all_locations = graph_db.locations

    # Step 0: Extract alphabetic words and remove generic stopwords ("port", "of", "in", "factory", etc.)
    # re.findall(...) splits on spaces/punctuation:
    #   Case A: ["port", "of", "san", "diego"] - LOCATION_STOPWORDS -> query_words = ["san", "diego"]
    #   Case B: ["ports", "in", "chna"]        - LOCATION_STOPWORDS -> query_words = ["chna"]
    query_words = [
        w for w in re.findall(r"[a-z\u00c0-\u017f]+", q_lower)
        if w not in LOCATION_STOPWORDS
    ]

    # Re-join the remaining geographic words into a clean search phrase
    #   Case A: clean_geo_query = "san diego"
    #   Case B: clean_geo_query = "chna"
    clean_geo_query = " ".join(query_words).strip() or q_lower

    # -------------------------------------------------------------------------
    # STEP 1: Exact whole-phrase match on Location.name or Location.country
    # -------------------------------------------------------------------------
    # Why `\b...\b` (word boundaries)?
    #   If clean_geo_query = "san diego", `\bsan diego\b` matches "san diego factory"
    #   but will NEVER match "busan".
    direct_matches = []
    for loc in all_locations:
        # Example L1: loc_name_l = "san diego factory", loc_country_l = "united states"
        loc_name_l = loc.name.lower()
        loc_country_l = loc.country.lower()
        if (
            re.search(rf"\b{re.escape(clean_geo_query)}\b", loc_name_l)      # Case A: "\bsan diego\b" matches "san diego factory" -> True!
            or re.search(rf"\b{re.escape(clean_geo_query)}\b", loc_country_l) # Matches if query was "United States" or "China"
            or loc_country_l in q_lower                                       # Matches if query was "all ports in china"
        ):
            direct_matches.append(loc)

    # Case A: direct_matches = [Location("San Diego Factory", "United States")] -> returns immediately!
    # Case B: direct_matches = [] (because "chna" is misspelled) -> continues to Step 2
    if direct_matches:
        return direct_matches, "direct_match"

    # -------------------------------------------------------------------------
    # STEP 2: Whole-word token match (handles multi-city queries like "Shanghai and Tokyo")
    # -------------------------------------------------------------------------
    # Keep only words with >= 4 letters so short prefixes like "san" (3 chars) cannot match alone
    #   Case B ("ports in Chna"): significant_tokens = ["chna"]
    #   Multi-city ("Shanghai and Tokyo"): significant_tokens = ["shanghai", "tokyo"]
    significant_tokens = [w for w in query_words if len(w) >= 4]
    token_matches = []
    for loc in all_locations:
        # Build a set of non-stopword words for this Location:
        #   For L2 ("Shanghai Port", "China"): loc_words = {"shanghai", "china"}
        #   For L3 ("Port of Busan", "South Korea"): loc_words = {"busan", "south", "korea"}
        loc_words = set(re.findall(r"[a-z\u00c0-\u017f]+", f"{loc.name} {loc.country}".lower())) - LOCATION_STOPWORDS

        # Check if any 4+ letter query word is an exact word in `loc_words`
        #   Multi-city: "shanghai" in {"shanghai", "china"} -> True!
        #   Case B:     "chna" in {"shanghai", "china"}     -> False
        if any(tok in loc_words for tok in significant_tokens):
            token_matches.append(loc)

    if token_matches:
        return token_matches, "token_match"

    # -------------------------------------------------------------------------
    # STEP 3: Fuzzy / Typo Resolution using Character Edit Similarity (>= 0.72)
    # -------------------------------------------------------------------------
    # Handles misspellings like:
    #   • "chna"     vs "china"    -> SequenceMatcher ratio = 2*4 / (4+5) = 0.89 (>= 0.72 -> Match!)
    #   • "shanghia" vs "shanghai" -> SequenceMatcher ratio = 2*7 / (8+8) = 0.88 (>= 0.72 -> Match!)
    #   • "shenzen"  vs "shenzhen" -> SequenceMatcher ratio = 2*7 / (7+8) = 0.93 (>= 0.72 -> Match!)
    fuzzy_matched_countries: set[str] = set()
    fuzzy_matched_locs: list[Location] = []
    for loc in all_locations:
        # For L2: loc_country_l = "china", loc_words = ["shanghai"]
        loc_country_l = loc.country.lower()
        loc_words = [
            w for w in re.findall(r"[a-z\u00c0-\u017f]+", loc.name.lower())
            if w not in LOCATION_STOPWORDS
        ]
        for q_tok in query_words:
            # 3a. Compare query token ("chna") against country name ("china")
            #     ratio("chna", "china") = 0.89 >= 0.72 -> adds "China" to fuzzy_matched_countries
            if len(q_tok) >= 4 and difflib.SequenceMatcher(None, q_tok, loc_country_l).ratio() >= 0.72:
                fuzzy_matched_countries.add(loc.country)

            # 3b. Compare query token ("shanghia") against each city word ("shanghai")
            #     ratio("shanghia", "shanghai") = 0.88 >= 0.72 -> appends L2 to fuzzy_matched_locs
            for lw in loc_words:
                if len(q_tok) >= 4 and difflib.SequenceMatcher(None, q_tok, lw).ratio() >= 0.72:
                    if loc not in fuzzy_matched_locs:
                        fuzzy_matched_locs.append(loc)

    # If a country typo matched (e.g. "Chna" -> {"China"}), return ALL locations in that country
    # Case B Output: ([Location("Shanghai Port", "China")], "fuzzy_country_match(China)")
    if fuzzy_matched_countries:
        country_locs = [loc for loc in all_locations if loc.country in fuzzy_matched_countries]
        return country_locs, f"fuzzy_country_match({', '.join(sorted(fuzzy_matched_countries))})"

    # Otherwise, if a city typo matched (e.g. "shanghia" -> [Location("Shanghai Port", "China")]), return those locations
    if fuzzy_matched_locs:
        return fuzzy_matched_locs, "fuzzy_city_match"

    # If nothing matched (e.g., querying a city that wasn't generated in this random run), return empty list
    return [], "no_match_in_graph"


# =====================================================================
# PHASE 1 COMPOSITE GRAPHRAG & DETERMINISTIC AGGREGATION TOOLS
# =====================================================================

def analyze_location_blast_radius(location_or_country_query: str, risk_threshold: float = 0.6) -> str:
    """
    Primary multi-hop GraphRAG tool for assessing supply chain disruptions at a location,
    city, port, or entire country (handles misspellings automatically).

    Traverses 2 hops in the knowledge graph:
      Location <-[:LOCATED_AT]- Supplier -[:MANUFACTURES]-> Product
    and deterministically calculates exact supplier revenue exposure, risk-weighted
    exposure, high-risk supplier counts, affected products, and alternative suppliers
    outside the disrupted zone.

    ===========================================================================
    WALKTHROUGH WITH 2 RUNNING TEST CASES
    ===========================================================================
    Assume `graph_db` has:
      • Disrupted Zone (San Diego / China):
        Location L1: Location("San Diego Factory", "United States")
          ├── Supplier S1: Supplier("Johnson PLC",  risk_score=0.73, revenue=100_000_000.0)
          │     └── Product P1: Product("Industrial Router 101", sku="E-101", price=500.0)
          └── Supplier S2: Supplier("Rogers-Davis", risk_score=0.20, revenue=50_000_000.0)
                └── Product P2: Product("Surgical Mask Carton 202", sku="P-202", price=100.0)
      • Outside Zone (Germany):
        Location L_other: Location("Frankfurt Warehouse", "Germany")
          └── Supplier S_alt: Supplier("Weber GmbH", risk_score=0.15, revenue=80_000_000.0)
                └── Product P_alt: Product("Industrial Router 999", sku="E-999", price=480.0)

    • Test Case 1: analyze_location_blast_radius("San Diego", risk_threshold=0.6)
      1. Resolves "San Diego" -> matched_locations = [L1]
      2. Hop 1 (Location -> Suppliers): finds S1 ("Johnson PLC") and S2 ("Rogers-Davis")
      3. Hop 2 (Suppliers -> Products): finds P1 ("Industrial Router 101") and P2 ("Surgical Mask Carton 202")
      4. Deterministic Math:
         - total_revenue        = 100M + 50M = $150,000,000.00
         - high_risk_sups (>0.6)= [S1] -> high_risk_revenue = $100,000,000.00
         - risk_weighted_rev    = (100M * 0.73) + (50M * 0.20) = $83,000,000.00
      5. Alternative Supplier Scan (Outside L1):
         - Extracts base product types at risk: {"Industrial Router", "Surgical Mask Carton"}
         - Scans L_other ("Frankfurt Warehouse"): finds S_alt ("Weber GmbH") also makes base type "Industrial Router" ("Industrial Router 999")!
         - Flags "Surgical Mask Carton" as sole-sourced (no external alternative).

    • Test Case 2: analyze_location_blast_radius("Atlantis", risk_threshold=0.6)
      1. Resolves "Atlantis" -> matched_locations = []
      2. Immediately returns JSON with `"status": "no_locations_found"` and list of valid locations.
    ===========================================================================

    Args:
        location_or_country_query: Location, city, port, or country name such as San Diego, Ningbo-Zhoushan, China, or Shanghai.
        risk_threshold: Risk score threshold above which a supplier is flagged HIGH RISK, defaults to 0.6.

    Returns:
        JSON string containing deterministic financial totals and the full 2-hop subgraph.
    """
    # -------------------------------------------------------------------------
    # STEP 1: Resolve the input string into Location node(s) in `graph_db`
    # -------------------------------------------------------------------------
    # Input:  location_or_country_query = "San Diego"
    # Output: matched_locations = [Location(name="San Diego Factory", country="United States")]
    #         resolution_method = "direct_match"
    matched_locations, resolution_method = _resolve_locations_hybrid(location_or_country_query)

    # Guard clause: If no Location matched (e.g. "Atlantis"), return available locations so LLM doesn't guess
    if not matched_locations:
        return json.dumps({
            "query": location_or_country_query,
            "status": "no_locations_found",
            "available_countries": sorted({loc.country for loc in graph_db.locations}),
            "available_locations": [f"{loc.name} ({loc.country})" for loc in graph_db.locations],
        }, indent=2)

    # Reference the underlying NetworkX DiGraph
    nx_graph = graph_db.graph

    # Containers to accumulate the 2-hop subgraph and deduplicate Suppliers/Products
    locations_breakdown = []                 # List of dicts (one per matched Location)
    unique_suppliers: dict[str, Supplier] = {}  # Key: supplier.name -> Value: Supplier object
    unique_products: dict[str, Product] = {}    # Key: product.sku   -> Value: Product object

    # -------------------------------------------------------------------------
    # STEP 2: 2-Hop Graph Traversal (Location -> Supplier -> Product)
    # -------------------------------------------------------------------------
    # Iterate over each affected Location (e.g., L1 = Location("San Diego Factory", "United States"))
    for loc in matched_locations:
        # HOP 1: Find all Supplier nodes connected to `loc` via (Supplier)-[:LOCATED_AT]->(Location)
        # Input:  loc = Location("San Diego Factory", "United States")
        # Output: connected_suppliers = [Supplier("Johnson PLC", 0.73, 100M), Supplier("Rogers-Davis", 0.20, 50M)]
        connected_suppliers = [
            node for node in nx.all_neighbors(nx_graph, loc)
            if isinstance(node, Supplier)
        ]

        sup_entries = []
        for sup in connected_suppliers:
            # Record supplier in `unique_suppliers` dict by name so we never double-count revenue
            # Output: unique_suppliers["Johnson PLC"] = Supplier("Johnson PLC", 0.73, 100_000_000.0)
            unique_suppliers[sup.name] = sup

            # HOP 2: Find all Product nodes connected to `sup` via (Supplier)-[:MANUFACTURES]->(Product)
            # Note: `nx.all_neighbors(nx_graph, sup)` returns BOTH its Location and its Products,
            #       so `isinstance(node, Product)` filters strictly to Product nodes!
            # Input:  sup = Supplier("Johnson PLC", 0.73, 100M)
            # Output: connected_products = [Product("Industrial Router 101", sku="E-101", price=500.0)]
            connected_products = [
                node for node in nx.all_neighbors(nx_graph, sup)
                if isinstance(node, Product)
            ]

            prod_entries = []
            for prod in connected_products:
                # Record product in `unique_products` keyed by SKU
                # Output: unique_products["E-101"] = Product("Industrial Router 101", "E-101", 500.0)
                unique_products[prod.sku] = prod

                # Format product dictionary for JSON output
                # Output: {"name": "Industrial Router 101", "sku": "E-101", "price": 500.0, "price_formatted": "$500.00"}
                prod_entries.append({
                    "name": prod.name,
                    "sku": prod.sku,
                    "price": prod.price,
                    "price_formatted": f"${prod.price:,.2f}",
                })

            # Format supplier dictionary (including its nested `products_manufactured` list)
            # For S1 (risk 0.73 > 0.6): risk_level = "HIGH_RISK"
            # For S2 (risk 0.20 <= 0.6): risk_level = "MODERATE_OR_LOW_RISK"
            sup_entries.append({
                "name": sup.name,
                "risk_score": sup.risk_score,
                "risk_level": "HIGH_RISK" if sup.risk_score > risk_threshold else "MODERATE_OR_LOW_RISK",
                "revenue": sup.revenue,
                "revenue_formatted": f"${sup.revenue:,.2f}",
                "products_manufactured": prod_entries,
            })

        # Append this location's complete supplier+product tree
        locations_breakdown.append({
            "location_name": loc.name,
            "country": loc.country,
            "supplier_count": len(sup_entries),
            "suppliers": sup_entries,
        })

    # -------------------------------------------------------------------------
    # STEP 3: Deterministic Python Financial & Risk Aggregations
    # -------------------------------------------------------------------------
    # Example with S1 ($100M, risk=0.73) and S2 ($50M, risk=0.20):
    #   • total_revenue         = 100_000_000.0 + 50_000_000.0 = 150_000_000.0
    total_revenue = round(sum(s.revenue for s in unique_suppliers.values()), 2)

    #   • risk_weighted_revenue = (100_000_000.0 * 0.73) + (50_000_000.0 * 0.20) = 83_000_000.0
    risk_weighted_revenue = round(
        sum(s.revenue * s.risk_score for s in unique_suppliers.values()), 2
    )

    #   • high_risk_sups        = [Supplier("Johnson PLC", 0.73, 100M)] (since 0.73 > 0.6)
    high_risk_sups = [
        s for s in unique_suppliers.values() if s.risk_score > risk_threshold
    ]

    #   • high_risk_revenue     = 100_000_000.0
    high_risk_revenue = round(sum(s.revenue for s in high_risk_sups), 2)

    # -------------------------------------------------------------------------
    # STEP 4: Scan Outside the Disrupted Zone for Alternative Suppliers
    # -------------------------------------------------------------------------
    # Convert matched_locations to a set for O(1) lookup: {Location("San Diego Factory", "United States")}
    matched_loc_set = set(matched_locations)

    # Strip the trailing random model number from each affected Product to get its catalog product type:
    #   Input:  prod.name = "Industrial Router 101" -> parts = ["Industrial", "Router", "101"]
    #   Check:  parts[-1].isdigit() is True ("101") -> base_type = "Industrial Router"
    #   Output: affected_base_types = {
    #               "Industrial Router": ["Industrial Router 101 (E-101)"],
    #               "Surgical Mask Carton": ["Surgical Mask Carton 202 (P-202)"]
    #           }
    affected_base_types: dict[str, list[str]] = {}
    for prod in unique_products.values():
        parts = prod.name.split()
        base_type = " ".join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() else prod.name
        affected_base_types.setdefault(base_type, []).append(f"{prod.name} ({prod.sku})")

    alternative_suppliers_by_product_type: dict[str, list[dict]] = {}
    unique_alt_suppliers: dict[str, Supplier] = {}

    # Loop through all OTHER locations in the world that are NOT disrupted
    for other_loc in graph_db.locations:
        # Skip disrupted locations (e.g. skip "San Diego Factory")
        if other_loc in matched_loc_set:
            continue

        # Find suppliers at this non-disrupted location (e.g. "Weber GmbH" at "Frankfurt Warehouse")
        other_sups = [
            n for n in nx.all_neighbors(nx_graph, other_loc)
            if isinstance(n, Supplier)
        ]
        for other_sup in other_sups:
            if other_sup.name in unique_suppliers:
                continue

            # Find products made by this external supplier (e.g. Product("Industrial Router 999", "E-999", 480.0))
            other_prods = [
                n for n in nx.all_neighbors(nx_graph, other_sup)
                if isinstance(n, Product)
            ]
            for other_prod in other_prods:
                # Extract base product type of external product:
                #   Input:  "Industrial Router 999" -> other_base = "Industrial Router"
                parts = other_prod.name.split()
                other_base = (
                    " ".join(parts[:-1])
                    if len(parts) > 1 and parts[-1].isdigit()
                    else other_prod.name
                )
                # Does this external supplier make a product type that is at risk in our disrupted zone?
                #   "Industrial Router" in {"Industrial Router", "Surgical Mask Carton"} -> True!
                if other_base in affected_base_types:
                    unique_alt_suppliers[other_sup.name] = other_sup
                    alternative_suppliers_by_product_type.setdefault(other_base, []).append({
                        "alternative_supplier_name": other_sup.name,           # "Weber GmbH"
                        "supplier_risk_score": other_sup.risk_score,           # 0.15
                        "supplier_risk_level": (
                            "HIGH_RISK" if other_sup.risk_score > risk_threshold else "MODERATE_OR_LOW_RISK"
                        ),                                                     # "MODERATE_OR_LOW_RISK"
                        "supplier_current_revenue": other_sup.revenue,         # 80_000_000.0
                        "supplier_current_revenue_formatted": f"${other_sup.revenue:,.2f}",  # "$80,000,000.00"
                        "location_name": other_loc.name,                       # "Frankfurt Warehouse"
                        "country": other_loc.country,                          # "Germany"
                        "equivalent_product": other_prod.name,                 # "Industrial Router 999"
                        "equivalent_sku": other_prod.sku,                      # "E-999"
                        "equivalent_unit_price_formatted": f"${other_prod.price:,.2f}",      # "$480.00"
                    })

    # Sum revenue of all unique external alternative suppliers found
    #   Output: total_alt_revenue = 80_000_000.0 ("Weber GmbH")
    total_alt_revenue = round(sum(s.revenue for s in unique_alt_suppliers.values()), 2)

    # Identify any affected product types that had ZERO external alternative suppliers in the graph
    #   Output: unmatched_product_types = ["Surgical Mask Carton"]
    unmatched_product_types = [
        bt for bt in affected_base_types if bt not in alternative_suppliers_by_product_type
    ]

    # -------------------------------------------------------------------------
    # STEP 5: Package Everything into a Single Structured JSON Payload
    # -------------------------------------------------------------------------
    report = {
        "query": location_or_country_query,
        "resolution_method": resolution_method,
        "risk_threshold_used": risk_threshold,
        "deterministic_summary_metrics": {
            "total_locations_affected": len(matched_locations),
            "total_suppliers_affected": len(unique_suppliers),
            "high_risk_supplier_count": len(high_risk_sups),
            "total_products_affected": len(unique_products),
            "total_supplier_revenue_exposed": total_revenue,
            "total_supplier_revenue_exposed_formatted": f"${total_revenue:,.2f}",
            "high_risk_supplier_revenue_exposed": high_risk_revenue,
            "high_risk_supplier_revenue_exposed_formatted": f"${high_risk_revenue:,.2f}",
            "risk_weighted_revenue_exposure": risk_weighted_revenue,
            "risk_weighted_revenue_exposure_formatted": f"${risk_weighted_revenue:,.2f}",
            "alternative_suppliers_outside_zone_count": len(unique_alt_suppliers),
            "total_alternative_suppliers_revenue_formatted": f"${total_alt_revenue:,.2f}",
        },
        "affected_locations": locations_breakdown,
        "alternative_suppliers_outside_disrupted_zone": {
            "by_product_type": alternative_suppliers_by_product_type,
            "product_types_with_no_external_alternative": unmatched_product_types,
        },
    }
    return json.dumps(report, indent=2)


def audit_products_and_upstream_suppliers(
    product_query: str,
    risk_threshold: float = 0.6,
    k: int = 25,
) -> str:
    """
    Primary multi-hop upstream GraphRAG tool for product or category risk audits
    (e.g., 'Pharmaceutical', 'medical products', 'electronics', 'syringes').

    Finds all matching products (combining category-catalog matching and semantic vector
    search up to k=25 so no products are truncated) and traverses 2 hops upstream:
      Product <-[:MANUFACTURES]- Supplier -[:LOCATED_AT]-> Location
    to identify manufacturing suppliers, their locations, and whether their risk_score
    exceeds `risk_threshold`.

    Args:
        product_query: Product name, description, or category such as Pharmaceutical, medical supplies, Electronics, or Brake Disc.
        risk_threshold: Risk score threshold to flag high-risk suppliers, defaults to 0.6.
        k: Max semantic search candidates if no exact category match, defaults to 25.

    Returns:
        JSON string with deterministic counts, high-risk supplier table, and full product mapping.
    """
    q_lower = product_query.lower().strip()
    matched_products: dict[str, Product] = {}

    # 1. Deterministic category / catalog keyword matching (prevents k=5 truncation!)
    matched_catalog_phrases: set[str] = set()
    for cat_key, phrases in CATEGORY_KEYWORDS.items():
        if cat_key in q_lower:
            matched_catalog_phrases.update(phrases)
        else:
            for phrase in phrases:
                if any(word in q_lower for word in phrase.split() if len(word) >= 4):
                    matched_catalog_phrases.add(phrase)

    if matched_catalog_phrases:
        for prod in graph_db.products:
            p_lower = prod.name.lower()
            if any(phrase in p_lower for phrase in matched_catalog_phrases):
                matched_products[prod.sku] = prod

    # 2. Direct substring match on product name or SKU
    for prod in graph_db.products:
        if q_lower in prod.name.lower() or q_lower == prod.sku.lower():
            matched_products[prod.sku] = prod

    # Guard: Before falling back to semantic search on products, check if the LLM
    # mistakenly passed a geographic city/country query (e.g., "Shanghai Shenzhen")
    # to this product tool instead of `analyze_location_blast_radius`.
    if not matched_products:
        loc_matches, loc_method = _resolve_locations_hybrid(product_query)
        if loc_matches and not loc_method.startswith("vector_semantic"):
            return json.dumps(
                {
                    "query": product_query,
                    "misrouted_location_query_blocked": True,
                    "message": (
                        f"'{product_query}' matches geographic location(s) "
                        f"({[l.name for l in loc_matches]}), not a product or product category. "
                        "Use `analyze_location_blast_radius` for locations/cities/countries."
                    ),
                },
                indent=2,
            )

    # 3. Fallback to ChromaDB semantic search if no catalog/substring matches found
    if not matched_products:
        sem_results = product_vector_db.get_products(
            product_query, k=min(k, max(1, len(graph_db.products)))
        )
        for prod in sem_results:
            matched_products[prod.sku] = prod

    nx_graph = graph_db.graph
    product_mappings = []
    suppliers_seen: dict[str, dict] = {}

    for prod in matched_products.values():
        # Find upstream supplier(s)
        prod_suppliers = [
            node for node in nx.all_neighbors(nx_graph, prod)
            if isinstance(node, Supplier)
        ]
        for sup in prod_suppliers:
            sup_locations = [
                node for node in nx.all_neighbors(nx_graph, sup)
                if isinstance(node, Location)
            ]
            loc = sup_locations[0] if sup_locations else Location(name="Unknown", country="Unknown")
            is_high_risk = sup.risk_score > risk_threshold

            product_mappings.append({
                "product_name": prod.name,
                "sku": prod.sku,
                "price": prod.price,
                "price_formatted": f"${prod.price:,.2f}",
                "supplier_name": sup.name,
                "supplier_risk_score": sup.risk_score,
                "supplier_risk_flag": "HIGH_RISK" if is_high_risk else "LOW_RISK",
                "supplier_revenue": sup.revenue,
                "supplier_revenue_formatted": f"${sup.revenue:,.2f}",
                "location_name": loc.name,
                "country": loc.country,
            })

            if sup.name not in suppliers_seen:
                suppliers_seen[sup.name] = {
                    "supplier_name": sup.name,
                    "risk_score": sup.risk_score,
                    "risk_flag": "HIGH_RISK" if is_high_risk else "LOW_RISK",
                    "revenue": sup.revenue,
                    "revenue_formatted": f"${sup.revenue:,.2f}",
                    "location": loc.name,
                    "country": loc.country,
                    "matched_products": [],
                }
            suppliers_seen[sup.name]["matched_products"].append(
                f"{prod.name} (SKU: {prod.sku}, ${prod.price:,.2f})"
            )

    # Sort product mappings and supplier summaries by risk_score descending
    product_mappings.sort(key=lambda x: x["supplier_risk_score"], reverse=True)
    all_suppliers_sorted = sorted(
        suppliers_seen.values(), key=lambda x: x["risk_score"], reverse=True
    )
    high_risk_suppliers = [s for s in all_suppliers_sorted if s["risk_flag"] == "HIGH_RISK"]
    low_risk_suppliers = [s for s in all_suppliers_sorted if s["risk_flag"] == "LOW_RISK"]

    total_sup_revenue = round(sum(s["revenue"] for s in all_suppliers_sorted), 2)
    high_risk_sup_revenue = round(sum(s["revenue"] for s in high_risk_suppliers), 2)

    report = {
        "query": product_query,
        "risk_threshold_used": risk_threshold,
        "deterministic_summary_metrics": {
            "total_matching_products": len(product_mappings),
            "total_unique_suppliers": len(all_suppliers_sorted),
            "high_risk_supplier_count": len(high_risk_suppliers),
            "low_risk_supplier_count": len(low_risk_suppliers),
            "total_supplier_revenue_exposed_formatted": f"${total_sup_revenue:,.2f}",
            "high_risk_supplier_revenue_exposed_formatted": f"${high_risk_sup_revenue:,.2f}",
        },
        "high_risk_suppliers": high_risk_suppliers,
        "low_risk_suppliers": low_risk_suppliers,
        "product_to_supplier_location_table": product_mappings,
    }
    return json.dumps(report, indent=2)


# =====================================================================
# GRANULAR FALLBACK TOOLS (retained for compatibility with baseline)
# =====================================================================

def retrieve_product_info(query: str):
    """Fallback semantic search for individual products (returns top 15 matches)."""
    products = product_vector_db.get_products(query, k=min(15, max(1, len(graph_db.products))))
    return_products = []
    for product in products:
        if query.lower() == product.name.lower():
            return [str(product)]
        return_products.append(str(product))
    return return_products


def retrieve_supplier_info(query: str):
    """Fallback semantic search for individual suppliers (returns top 10 matches)."""
    suppliers = supplier_vector_db.get_suppliers(query, k=min(10, max(1, len(graph_db.suppliers))))
    if not suppliers:
        return [f"No suppliers found matching '{query}'. Try a different search term."]
    return_suppliers = []
    for supplier in suppliers:
        if query.lower() == supplier.name.lower():
            return [str(supplier)]
        return_suppliers.append(str(supplier))
    return return_suppliers


def retrieve_location_info(query: str):
    """Fallback semantic search for individual locations (returns top 10 matches)."""
    locations = location_vector_db.get_locations(query, k=min(10, max(1, len(graph_db.locations))))
    if not locations:
        return [f"No locations found matching '{query}'. Try a different search term."]
    return_locations = []
    for location in locations:
        if query.lower() == location.name.lower():
            return [str(location)]
        return_locations.append(str(location))
    return return_locations


def explore_graph_connections(node_name: str):
    """Fallback 1-hop graph neighbor lookup for a specific exact node name."""
    found_node = graph_db.get_node_by_name(node_name)
    if not found_node:
        return [
            f"No node found matching '{node_name}'. "
            f"Prefer calling analyze_location_blast_radius('{node_name}') for locations/countries "
            f"or audit_products_and_upstream_suppliers('{node_name}') for products."
        ]
    neighbors = list(nx.all_neighbors(graph_db.graph, found_node))
    if not neighbors:
        return [f"Node '{found_node.name}' found but has no connections in the graph."]
    return [str(neighbor) for neighbor in neighbors]


def launch_risk_map_ui(risk_threshold: float = 0.6):
    """Launch the Streamlit visualization app using a DOT snapshot of `graph_db`."""
    try:
        dot = build_supply_chain_graphviz(graph_db, risk_threshold=risk_threshold)
        dot_file = os.path.join(tempfile.gettempdir(), "supply_chain_graph.dot")
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot.source)

        env = os.environ.copy()
        env["SC_GRAPH_DOT_PATH"] = dot_file
        env["SC_GRAPH_THRESHOLD"] = str(risk_threshold)

        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / "app.py")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return (
            "Launched Global Supply Chain Risk Map at http://localhost:8501 "
            f"(rendering graph snapshot from {dot_file})"
        )
    except Exception as e:
        return f"Failed to launch Streamlit app: {e}"


# =====================================================================
# PHASE 2: STRUCTURED AGENT STATE & STATE-AWARE TOOL NODE
# =====================================================================

class AgentState(TypedDict, total=False):
    """
    Phase 2 Structured Working Memory for the LangGraph Supply Chain Agent.

    Instead of relying only on a flat `messages` list (where consecutive OpenRouter
    free-tier models must re-parse raw `ToolMessage` JSON strings from scratch),
    `AgentState` maintains canonical, deduplicated entities and Python-computed
    metrics across turns:
      - `messages`: Full conversation history (`HumanMessage`, `AIMessage`, `ToolMessage`).
      - `visited_nodes`: Canonical tool call signatures (`tool_name::normalized_args`)
        and traversed node names so duplicate tool calls are intercepted immediately.
      - `discovered_locations`: Canonical Location dicts keyed/deduplicated by `location_name`.
      - `discovered_suppliers`: Canonical Supplier dicts keyed/deduplicated by `name`.
      - `discovered_products`: Canonical Product dicts keyed/deduplicated by `sku`.
      - `deterministic_metrics`: Latest Python-computed financial and count rollups.
      - `reflection_feedback`: Optional critique string (prepared for Phase 3 CRAG).
      - `reflection_count`: Number of verification/reflection retries performed.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    visited_nodes: list[str]
    discovered_locations: list[dict]
    discovered_suppliers: list[dict]
    discovered_products: list[dict]
    deterministic_metrics: dict
    reflection_feedback: str | None
    reflection_count: int


tools = [
    analyze_location_blast_radius,
    audit_products_and_upstream_suppliers,
    retrieve_product_info,
    retrieve_supplier_info,
    retrieve_location_info,
    explore_graph_connections,
]
TOOL_MAP = {t.__name__: t for t in tools}


def _make_call_signature(tool_name: str, tool_args: dict) -> str:
    """
    Create a canonical, case- and key-order-insensitive signature for a tool call
    so `stateful_tool_node` can detect duplicate invocations across turns.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Example A (Location Blast Radius tool call with messy spacing/casing):
      - Input:
          `tool_name = "analyze_location_blast_radius"`
          `tool_args = {"risk_threshold": 0.6, "location_or_country_query": "  San Diego  "}`
      - Output:
          `"analyze_location_blast_radius::{\"location_or_country_query\": \"san diego\", \"risk_threshold\": 0.6}"`
        (Notice keys are sorted alphabetically and `"  San Diego  "` is normalized to `"san diego"`,
         so if a second model calls `{"location_or_country_query": "san diego", "risk_threshold": 0.6}`,
         it produces the EXACT same signature string and gets blocked as a duplicate!)

    Example B (1-Hop Graph Exploration tool call):
      - Input:
          `tool_name = "explore_graph_connections"`
          `tool_args = {"node_name": "Johnson PLC"}`
      - Output:
          `"explore_graph_connections::{\"node_name\": \"johnson plc\"}"`
    ===========================================================================
    """
    # Normalize every argument key/value so key order and string casing/whitespace do not matter:
    #   - `(tool_args or {}).items()`:
    #       Ex A Input: `{"risk_threshold": 0.6, "location_or_country_query": "  San Diego  "}`
    #       Ex B Input: `{"node_name": "Johnson PLC"}`
    #   - `sorted((tool_args or {}).items())` sorts tuples by parameter name (`k`):
    #       Ex A -> `[("location_or_country_query", "  San Diego  "), ("risk_threshold", 0.6)]`
    #       Ex B -> `[("node_name", "Johnson PLC")]`
    #   - `(v.strip().lower() if isinstance(v, str) else v)` cleans string values and keeps numbers as-is:
    #       Ex A -> `"  San Diego  "` becomes `"san diego"`; `0.6` stays `0.6`
    #       Ex B -> `"Johnson PLC"` becomes `"johnson plc"`
    #   - Final `normalized_args` dict:
    #       Ex A Output: `{"location_or_country_query": "san diego", "risk_threshold": 0.6}`
    #       Ex B Output: `{"node_name": "johnson plc"}`
    normalized_args = {
        k: (v.strip().lower() if isinstance(v, str) else v)
        for k, v in sorted((tool_args or {}).items())
    }

    # Combine `tool_name`, `"::"`, and deterministic JSON string (`json.dumps(..., sort_keys=True)`):
    #   - Ex A Output: `'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}'`
    #   - Ex B Output: `'explore_graph_connections::{"node_name": "johnson plc"}'`
    return f"{tool_name}::{json.dumps(normalized_args, sort_keys=True)}"


def _ingest_structured_tool_output(
    raw_output: str,
    visited: set[str],
    locations_by_name: dict[str, dict],
    suppliers_by_name: dict[str, dict],
    products_by_sku: dict[str, dict],
    metrics_store: dict,
) -> None:
    """
    Parse structured JSON returned by `analyze_location_blast_radius` or
    `audit_products_and_upstream_suppliers` and merge canonical entities +
    deterministic metrics in-place into the Phase 2 `AgentState` dictionaries.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Initial mutable containers before call (empty at Turn 1):
      `visited = {"analyze_location_blast_radius::{\"location_or_country_query\": \"san diego\", \"risk_threshold\": 0.6}"}`
      `locations_by_name = {}`
      `suppliers_by_name = {}`
      `products_by_sku   = {}`
      `metrics_store     = {}`

    Example A (`raw_output` from `analyze_location_blast_radius("San Diego", 0.6)`):
      - Input `raw_output`: JSON string containing:
        ```json
        {
          "deterministic_summary_metrics": {
            "total_locations_affected": 1,
            "total_suppliers_affected": 2,
            "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"
          },
          "affected_locations": [
            {
              "location_name": "San Diego Factory",
              "country": "USA",
              "supplier_count": 2,
              "high_risk_supplier_count": 1,
              "location_supplier_revenue_formatted": "$1,113,139,177.74",
              "suppliers": [
                {
                  "name": "Johnson PLC",
                  "location": "San Diego Factory",
                  "risk_score": 0.73,
                  "risk_level": "HIGH_RISK",
                  "revenue": 124049854.5,
                  "revenue_formatted": "$124,049,854.50",
                  "product_count": 1,
                  "products_manufactured": [
                    {
                      "name": "Grid Monitoring Meter 9971",
                      "sku": "E-53774",
                      "price": 2028.2,
                      "price_formatted": "$2,028.20"
                    }
                  ]
                }
              ]
            }
          ]
        }
        ```
      - Resulting in-place mutations after Example A finishes:
        * `metrics_store`     -> `{"total_locations_affected": 1, "total_suppliers_affected": 2, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"}`
        * `locations_by_name` -> `{"San Diego Factory": {"location_name": "San Diego Factory", "country": "USA", ...}}`
        * `suppliers_by_name` -> `{"Johnson PLC": {"name": "Johnson PLC", "risk_score": 0.73, ...}}`
        * `products_by_sku`   -> `{"E-53774": {"name": "Grid Monitoring Meter 9971", "sku": "E-53774", ...}}`
        * `visited` adds      -> `"node::san diego factory"`, `"node::johnson plc"`, `"node::grid monitoring meter 9971"`

    Example B (`raw_output` from `audit_products_and_upstream_suppliers("Pharmaceutical", 0.6)`):
      - Input `raw_output`: JSON string containing `"deterministic_summary_metrics"` and
        `"product_to_supplier_location_table": [{"product_name": "Sterile Syringe Kit 104", "sku": "P-10400", "price": 450.0, "price_formatted": "$450.00", "supplier_name": "MedCore Ltd", "supplier_risk_score": 0.81, "supplier_risk_flag": "HIGH_RISK", "supplier_revenue": 50000000.0, "supplier_revenue_formatted": "$50,000,000.00", "location_name": "Basel Plant", "country": "Switzerland"}]`
      - Resulting in-place mutations after Example B finishes:
        * `locations_by_name["Basel Plant"]`, `suppliers_by_name["MedCore Ltd"]`, `products_by_sku["P-10400"]` added,
          and `"node::basel plant"`, `"node::medcore ltd"`, `"node::sterile syringe kit 104"` added to `visited`.
    ===========================================================================
    """
    # Guard 1: Check that `raw_output` is actually a string.
    #   - Ex A & Ex B Input: `isinstance("{...}", str)` -> Output: `True` (does not return early)
    if not isinstance(raw_output, str):
        return

    # Strip leading/trailing whitespace/newlines so we can inspect the first and last characters.
    #   - Ex A & Ex B Input: `'{\n  "query": ... \n}'` -> Output `stripped`: `'{"query": ...}'`
    stripped = raw_output.strip()

    # Guard 2: Only parse JSON object strings (starts with `{` and ends with `}`).
    # If a 1-hop fallback tool returned a Python list string like `"['Supplier: ...']"`, skip JSON ingestion.
    #   - Ex A & Ex B: `stripped.startswith("{") and stripped.endswith("}")` -> `True`
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return

    # Parse the JSON string into a Python dictionary `payload`:
    #   - Ex A Output `payload`: `{"deterministic_summary_metrics": {...}, "affected_locations": [...]}`
    #   - Ex B Output `payload`: `{"deterministic_summary_metrics": {...}, "product_to_supplier_location_table": [...]}`
    try:
        payload = json.loads(stripped)
    except Exception:
        return

    # -------------------------------------------------------------------------
    # STEP 1: Preserve per-query deterministic metrics without overwriting
    #         previous queries, then recompute exact cumulative metrics in STEP 4
    # -------------------------------------------------------------------------
    # Look up `"deterministic_summary_metrics"` and `"query"` from the parsed JSON dict:
    #   - Ex A Output `det_metrics`: `{"total_locations_affected": 1, "total_suppliers_affected": 2, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"}`
    #   - Ex B Output `det_metrics`: `{"total_matching_products": 1, "high_risk_supplier_count": 1, ...}`
    det_metrics = payload.get("deterministic_summary_metrics")
    query_label = str(payload.get("query") or "latest_query").strip()

    # Store each tool call's own summary inside `metrics_store["per_query_breakdown"][query_label]`
    # so a second tool call (e.g., "Rotterdam" after "San Diego") NEVER wipes out the first call's metrics:
    if isinstance(det_metrics, dict) and det_metrics:
        per_query = dict(metrics_store.get("per_query_breakdown") or {})
        per_query[query_label] = det_metrics
        metrics_store["per_query_breakdown"] = per_query

    # -------------------------------------------------------------------------
    # STEP 2: Extract entities from `analyze_location_blast_radius` output
    #         (Active in Example A; skipped in Example B because key is absent)
    # -------------------------------------------------------------------------
    # Iterate over `payload.get("affected_locations")`:
    #   - Ex A: loops over 1 location dict (`loc = {"location_name": "San Diego Factory", "country": "USA", ...}`)
    #   - Ex B: `payload.get("affected_locations")` is `None` -> `or []` yields `[]` (0 iterations)
    for loc in payload.get("affected_locations") or []:
        # Extract location name:
        #   - Ex A: `loc_name = "San Diego Factory"`
        loc_name = loc.get("location_name")
        if loc_name:
            # Record `"node::san diego factory"` in `visited` so future 1-hop calls like
            # `explore_graph_connections("San Diego Factory")` are blocked as redundant:
            visited.add(f"node::{loc_name.lower()}")
            # Save canonical location record keyed by `"San Diego Factory"`:
            #   - Ex A Output `locations_by_name["San Diego Factory"]`:
            #     `{"location_name": "San Diego Factory", "country": "USA", "supplier_count": 2, "high_risk_supplier_count": 1, "location_supplier_revenue_formatted": "$1,113,139,177.74"}`
            locations_by_name[loc_name] = {
                "location_name": loc_name,
                "country": loc.get("country", "Unknown"),
                "supplier_count": loc.get("supplier_count", 0),
                "high_risk_supplier_count": loc.get("high_risk_supplier_count", 0),
                "location_supplier_revenue_formatted": loc.get("location_supplier_revenue_formatted", "$0.00"),
            }

        # Loop over every supplier inside this location (`loc.get("suppliers")`):
        #   - Ex A: `sup = {"name": "Johnson PLC", "risk_score": 0.73, "risk_level": "HIGH_RISK", ...}`
        for sup in loc.get("suppliers") or []:
            # Extract supplier name:
            #   - Ex A: `sup_name = "Johnson PLC"`
            sup_name = sup.get("name")
            if sup_name:
                # Mark `"node::johnson plc"` as visited in `visited`:
                visited.add(f"node::{sup_name.lower()}")
                # Store canonical supplier dict keyed by `"Johnson PLC"`:
                #   - Ex A Output `suppliers_by_name["Johnson PLC"]`:
                #     `{"name": "Johnson PLC", "location_name": "San Diego Factory", "country": "USA", "risk_score": 0.73, "risk_level": "HIGH_RISK", "revenue": 124049854.5, "revenue_formatted": "$124,049,854.50", "product_count": 1}`
                suppliers_by_name[sup_name] = {
                    "name": sup_name,
                    "location_name": loc_name or sup.get("location", "Unknown"),
                    "country": loc.get("country", "Unknown"),
                    "risk_score": sup.get("risk_score"),
                    "risk_level": sup.get("risk_level"),
                    "revenue": sup.get("revenue"),
                    "revenue_formatted": sup.get("revenue_formatted"),
                    "product_count": sup.get("product_count", 0),
                }

            # Loop over every downstream product manufactured by `sup`:
            #   - Ex A: `prod = {"name": "Grid Monitoring Meter 9971", "sku": "E-53774", "price": 2028.2, "price_formatted": "$2,028.20"}`
            for prod in sup.get("products_manufactured") or []:
                # Extract unique product SKU:
                #   - Ex A: `sku = "E-53774"`
                sku = prod.get("sku")
                if sku:
                    # Mark `"node::grid monitoring meter 9971"` as visited:
                    visited.add(f"node::{ prod.get('name', '').lower() }")
                    # Store canonical product dict keyed by `"E-53774"`:
                    #   - Ex A Output `products_by_sku["E-53774"]`:
                    #     `{"name": "Grid Monitoring Meter 9971", "sku": "E-53774", "price": 2028.2, "price_formatted": "$2,028.20", "supplier_name": "Johnson PLC", "location_name": "San Diego Factory", "country": "USA"}`
                    products_by_sku[sku] = {
                        "name": prod.get("name"),
                        "sku": sku,
                        "price": prod.get("price"),
                        "price_formatted": prod.get("price_formatted"),
                        "supplier_name": sup_name,
                        "location_name": loc_name,
                        "country": loc.get("country", "Unknown"),
                    }

    # -------------------------------------------------------------------------
    # STEP 3: Extract entities from `audit_products_and_upstream_suppliers` output
    #         (Skipped in Example A; active in Example B)
    # -------------------------------------------------------------------------
    # Iterate over `payload.get("product_to_supplier_location_table")`:
    #   - Ex A: `None or []` -> `[]` (0 iterations)
    #   - Ex B: loops over `row = {"product_name": "Sterile Syringe Kit 104", "sku": "P-10400", "supplier_name": "MedCore Ltd", "location_name": "Basel Plant", "country": "Switzerland", ...}`
    for row in payload.get("product_to_supplier_location_table") or []:
        # Unpack row fields for Example B:
        #   - `loc_name  = "Basel Plant"`
        #   - `country   = "Switzerland"`
        #   - `sup_name  = "MedCore Ltd"`
        #   - `sku       = "P-10400"`
        #   - `prod_name = "Sterile Syringe Kit 104"`
        loc_name = row.get("location_name")
        country = row.get("country", "Unknown")
        sup_name = row.get("supplier_name")
        sku = row.get("sku")
        prod_name = row.get("product_name")

        # Add location to `visited` and `locations_by_name` if not already present:
        #   - Ex B Output: `visited` adds `"node::basel plant"`;
        #     `locations_by_name["Basel Plant"] = {"location_name": "Basel Plant", "country": "Switzerland"}`
        if loc_name and loc_name not in locations_by_name:
            visited.add(f"node::{loc_name.lower()}")
            locations_by_name[loc_name] = {
                "location_name": loc_name,
                "country": country,
            }

        # Add supplier to `visited` and `suppliers_by_name`:
        #   - Ex B Output: `visited` adds `"node::medcore ltd"`;
        #     `suppliers_by_name["MedCore Ltd"] = {"name": "MedCore Ltd", "location_name": "Basel Plant", "country": "Switzerland", "risk_score": 0.81, "risk_level": "HIGH_RISK", "revenue": 50000000.0, "revenue_formatted": "$50,000,000.00"}`
        if sup_name:
            visited.add(f"node::{sup_name.lower()}")
            suppliers_by_name[sup_name] = {
                "name": sup_name,
                "location_name": loc_name,
                "country": country,
                "risk_score": row.get("supplier_risk_score"),
                "risk_level": row.get("supplier_risk_flag"),
                "revenue": row.get("supplier_revenue"),
                "revenue_formatted": row.get("supplier_revenue_formatted"),
            }

        # Add product to `visited` and `products_by_sku`:
        #   - Ex B Output: `visited` adds `"node::sterile syringe kit 104"`;
        #     `products_by_sku["P-10400"] = {"name": "Sterile Syringe Kit 104", "sku": "P-10400", "price": 450.0, "price_formatted": "$450.00", "supplier_name": "MedCore Ltd", "location_name": "Basel Plant", "country": "Switzerland"}`
        if sku:
            if prod_name:
                visited.add(f"node::{prod_name.lower()}")
            products_by_sku[sku] = {
                "name": prod_name,
                "sku": sku,
                "price": row.get("price"),
                "price_formatted": row.get("price_formatted"),
                "supplier_name": sup_name,
                "location_name": loc_name,
                "country": country,
            }

    # -------------------------------------------------------------------------
    # STEP 4: Recompute Exact Cumulative Metrics Across ALL Discovered Entities
    #         (Fixes the `.update()` overwrite bug when 2+ tool calls run!)
    # -------------------------------------------------------------------------
    # Because `locations_by_name`, `suppliers_by_name`, and `products_by_sku` hold the
    # deduplicated union across ALL tool calls so far, recomputing totals directly from
    # them guarantees `metrics_store` always matches the exact union of discovered entities:
    if locations_by_name or suppliers_by_name or products_by_sku:
        all_sups = list(suppliers_by_name.values())
        high_risk_sups = [
            s for s in all_sups if s.get("risk_level") == "HIGH_RISK"
        ]
        low_risk_sups = [
            s for s in all_sups if s.get("risk_level") != "HIGH_RISK"
        ]
        total_rev = round(
            sum(float(s.get("revenue") or 0.0) for s in all_sups), 2
        )
        high_risk_rev = round(
            sum(float(s.get("revenue") or 0.0) for s in high_risk_sups), 2
        )
        risk_weighted_rev = round(
            sum(
                float(s.get("revenue") or 0.0) * float(s.get("risk_score") or 0.0)
                for s in all_sups
            ),
            2,
        )
        per_query_saved = metrics_store.get("per_query_breakdown")
        metrics_store.update(
            {
                "total_locations_affected": len(locations_by_name),
                "total_suppliers_affected": len(all_sups),
                "high_risk_supplier_count": len(high_risk_sups),
                "low_risk_supplier_count": len(low_risk_sups),
                "total_products_affected": len(products_by_sku),
                "total_supplier_revenue_exposed": total_rev,
                "total_supplier_revenue_exposed_formatted": f"${total_rev:,.2f}",
                "high_risk_supplier_revenue_exposed": high_risk_rev,
                "high_risk_supplier_revenue_exposed_formatted": f"${high_risk_rev:,.2f}",
                "risk_weighted_revenue_exposure": risk_weighted_rev,
                "risk_weighted_revenue_exposure_formatted": f"${risk_weighted_rev:,.2f}",
            }
        )
        if per_query_saved:
            metrics_store["per_query_breakdown"] = per_query_saved


def stateful_tool_node(state: AgentState) -> dict:
    """
    Phase 2 State-Aware Tool Execution Node (replaces generic `ToolNode(tools)`).

    1. Deduplication Guard: Checks `visited_nodes` before executing any tool call.
       - If the exact same tool call (`tool_name::args`) was already executed, OR
       - If a granular 1-hop tool (`explore_graph_connections`, `retrieve_*`) is called
         on an entity (`node::<name>`) already resolved in `AgentState`,
       it short-circuits and returns a fast Working-Memory cache notice instead of
       wasting turns re-querying ChromaDB/NetworkX.
    2. Structured State Extraction: Parses JSON outputs from composite GraphRAG tools
       and populates `discovered_locations`, `discovered_suppliers`, `discovered_products`,
       and `deterministic_metrics` in `AgentState`.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Example A (Turn 1 — Fresh composite tool call that executes & populates `AgentState`):
      - Input `state`:
          ```python
          {
              "messages": [
                  HumanMessage(content="Assess strike at San Diego"),
                  AIMessage(
                      content="",
                      tool_calls=[
                          {
                              "id": "call_1",
                              "name": "analyze_location_blast_radius",
                              "args": {"location_or_country_query": "San Diego", "risk_threshold": 0.6},
                          }
                      ],
                  ),
              ],
              # All Phase 2 memory fields are empty on Turn 1:
              "visited_nodes": [],
              "discovered_locations": [],
              "discovered_suppliers": [],
              "discovered_products": [],
              "deterministic_metrics": {},
          }
          ```
      - Execution Flow:
          1. Computes `signature = 'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}'`.
          2. Check 1 (`signature in visited`) -> `False`.
          3. Check 2 (`redundant_1hop_call_blocked`) -> `False` (since `tc_name` is a composite tool, not a 1-hop tool).
          4. Runs `analyze_location_blast_radius(location_or_country_query="San Diego", risk_threshold=0.6)`.
          5. Calls `_ingest_structured_tool_output(...)` which extracts `"San Diego Factory"`, `"Johnson PLC"`,
             `"Grid Monitoring Meter 9971"`, and `deterministic_summary_metrics` into `AgentState`.
      - Output:
          Returns updated `AgentState` dict with `ToolMessage(content='{"query": "San Diego", ...}')` and
          populated `visited_nodes`, `discovered_locations`, `discovered_suppliers`, `discovered_products`,
          and `deterministic_metrics`.

    Example B (Turn 2 — Next LLM turn tries TWO redundant tool calls in parallel):
      Suppose `openrouter/free` routes Turn 2 to a weaker model that ignores the Working Memory
      and dispatches 2 redundant tool calls in a single `AIMessage`:
        - Call B1 (`call_2`): `analyze_location_blast_radius({"location_or_country_query": "san diego", "risk_threshold": 0.6})`
          (Exact duplicate of Turn 1!)
        - Call B2 (`call_3`): `explore_graph_connections({"node_name": "Johnson PLC"})`
          (1-hop lookup on a supplier already discovered inside Turn 1's 2-hop subgraph!)
      - Input `state`:
          Contains the `visited_nodes`, `discovered_locations`, `discovered_suppliers`, `discovered_products`,
          and `deterministic_metrics` produced by Example A above, plus the new `AIMessage(tool_calls=[call_2, call_3])`.
      - Execution Flow:
          - For Call B1 (`call_2`): Hits **Check 1** (`signature in visited` is `True`) -> immediately
            appends a `"duplicate_call_blocked": true` `ToolMessage` and `continue`s without running the tool!
          - For Call B2 (`call_3`): Hits **Check 2** (`"node::johnson plc" in visited` is `True`) -> immediately
            appends a `"redundant_1hop_call_blocked": true` `ToolMessage` and `continue`s without touching NetworkX!
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # STEP 1: Unpack the latest `AIMessage` and its `tool_calls` from `state`
    # -------------------------------------------------------------------------
    # Get the conversation history list:
    #   - Ex A: `[HumanMessage(...), AIMessage(tool_calls=[call_1])]` (len = 2)
    #   - Ex B: `[HumanMessage(...), AIMessage(call_1), ToolMessage(...), AIMessage(tool_calls=[call_2, call_3])]` (len = 4)
    messages = state.get("messages", [])

    # Grab the last message in the list (which is the `AIMessage` that triggered `stateful_tool_node`):
    #   - Ex A: `last_msg = AIMessage(tool_calls=[call_1])`
    #   - Ex B: `last_msg = AIMessage(tool_calls=[call_2, call_3])`
    last_msg = messages[-1] if messages else None

    # Extract the list of tool call dicts from `last_msg.tool_calls`:
    #   - Ex A: `tool_calls = [{"id": "call_1", "name": "analyze_location_blast_radius", "args": {"location_or_country_query": "San Diego", "risk_threshold": 0.6}}]`
    #   - Ex B: `tool_calls = [{"id": "call_2", "name": "analyze_location_blast_radius", ...}, {"id": "call_3", "name": "explore_graph_connections", "args": {"node_name": "Johnson PLC"}}]`
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    # -------------------------------------------------------------------------
    # STEP 2: Rehydrate cumulative Phase 2 Working Memory from `state` into
    #         fast lookup sets/dicts keyed by canonical name or SKU
    # -------------------------------------------------------------------------
    # Convert `state["visited_nodes"]` list into a Python `set` for O(1) lookup:
    #   - Ex A (Turn 1 start): `visited = set()`
    #   - Ex B (Turn 2 start): `visited = {'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}', 'node::san diego factory', 'node::johnson plc', 'node::grid monitoring meter 9971'}`
    visited: set[str] = set(state.get("visited_nodes") or [])

    # Rebuild `locations_by_name` dict (`{location_name: loc_dict}`) from `state["discovered_locations"]`:
    #   - Ex A (Turn 1 start): `locations_by_name = {}`
    #   - Ex B (Turn 2 start): `locations_by_name = {"San Diego Factory": {"location_name": "San Diego Factory", "country": "USA", ...}}`
    locations_by_name: dict[str, dict] = {
        loc["location_name"]: loc
        for loc in (state.get("discovered_locations") or [])
        if "location_name" in loc
    }

    # Rebuild `suppliers_by_name` dict (`{supplier_name: sup_dict}`) from `state["discovered_suppliers"]`:
    #   - Ex A (Turn 1 start): `suppliers_by_name = {}`
    #   - Ex B (Turn 2 start): `suppliers_by_name = {"Johnson PLC": {"name": "Johnson PLC", "risk_score": 0.73, ...}}`
    suppliers_by_name: dict[str, dict] = {
        sup["name"]: sup
        for sup in (state.get("discovered_suppliers") or [])
        if "name" in sup
    }

    # Rebuild `products_by_sku` dict (`{sku: prod_dict}`) from `state["discovered_products"]`:
    #   - Ex A (Turn 1 start): `products_by_sku = {}`
    #   - Ex B (Turn 2 start): `products_by_sku = {"E-53774": {"name": "Grid Monitoring Meter 9971", "sku": "E-53774", ...}}`
    products_by_sku: dict[str, dict] = {
        prod["sku"]: prod
        for prod in (state.get("discovered_products") or [])
        if "sku" in prod
    }

    # Copy `state["deterministic_metrics"]` into a mutable dict `metrics_store`:
    #   - Ex A (Turn 1 start): `metrics_store = {}`
    #   - Ex B (Turn 2 start): `metrics_store = {"total_locations_affected": 1, "total_suppliers_affected": 2, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"}`
    metrics_store: dict = dict(state.get("deterministic_metrics") or {})

    # Initialize list of `ToolMessage` objects to return for this turn:
    #   - Ex A & Ex B: `tool_messages = []`
    tool_messages: list[ToolMessage] = []

    # -------------------------------------------------------------------------
    # STEP 3: Loop over each requested tool call in `tool_calls`
    # -------------------------------------------------------------------------
    for tc in tool_calls:
        # Unpack tool call ID, tool function name, and arguments dict:
        #   - Ex A (iter 1): `tc_id = "call_1"`, `tc_name = "analyze_location_blast_radius"`, `tc_args = {"location_or_country_query": "San Diego", "risk_threshold": 0.6}`
        #   - Ex B (iter 1 - Call B1): `tc_id = "call_2"`, `tc_name = "analyze_location_blast_radius"`, `tc_args = {"location_or_country_query": "san diego", "risk_threshold": 0.6}`
        #   - Ex B (iter 2 - Call B2): `tc_id = "call_3"`, `tc_name = "explore_graph_connections"`, `tc_args = {"node_name": "Johnson PLC"}`
        tc_id = tc.get("id", "unknown_call_id")
        tc_name = tc.get("name", "")
        tc_args = tc.get("args", {}) or {}

        # Build canonical call signature via `_make_call_signature`:
        #   - Ex A (iter 1): `signature = 'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}'`
        #   - Ex B (Call B1): `signature = 'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}'`
        #   - Ex B (Call B2): `signature = 'explore_graph_connections::{"node_name": "johnson plc"}'`
        signature = _make_call_signature(tc_name, tc_args)

        # ---------------------------------------------------------------------
        # CHECK 1: Exact Duplicate Tool Call Signature Guard
        # ---------------------------------------------------------------------
        # Check if `signature` is already in `visited`:
        #   - Ex A (iter 1): `signature in visited` -> `False` (proceeds to Check 2)
        #   - Ex B (Call B1): `signature in visited` -> `True`! (Enters `if` block below)
        #   - Ex B (Call B2): `signature in visited` -> `False` (proceeds to Check 2)
        if signature in visited:
            # Ex B (Call B1): Build a short-circuit JSON message telling the LLM this exact call
            # already ran and echoing `metrics_store` so it can write its report immediately:
            cached_summary = json.dumps(
                {
                    "duplicate_call_blocked": True,
                    "message": (
                        f"Tool call `{tc_name}({tc_args})` was ALREADY executed in this session. "
                        "Use the entities and `deterministic_metrics` already stored in AgentState "
                        "working memory to write your final report immediately."
                    ),
                    "deterministic_summary_metrics": metrics_store,
                },
                indent=2,
            )
            # Ex B (Call B1): Append `ToolMessage(tool_call_id="call_2", ...)` and skip execution via `continue`:
            tool_messages.append(
                ToolMessage(content=cached_summary, name=tc_name, tool_call_id=tc_id)
            )
            continue

        # ---------------------------------------------------------------------
        # CHECK 2: Redundant 1-Hop Call on an Entity Already in Working Memory
        # ---------------------------------------------------------------------
        # Extract the target entity string if the tool uses `node_name` or `query`:
        #   - Ex A (iter 1): `tc_args.get("node_name") or tc_args.get("query")` is `None` -> `granular_target = ""`
        #   - Ex B (Call B2): `tc_args.get("node_name")` is `"Johnson PLC"` -> `granular_target = "johnson plc"`
        granular_target = (
            tc_args.get("node_name") or tc_args.get("query") or ""
        ).strip().lower()

        # Check if this is a 1-hop tool AND `f"node::{granular_target}"` was already visited:
        #   - Ex A (iter 1): `tc_name` is `"analyze_location_blast_radius"` (not in 1-hop set) -> `False`
        #   - Ex B (Call B2):
        #       1. `tc_name in {"explore_graph_connections", ...}` -> `True`
        #       2. `granular_target` (`"johnson plc"`) is non-empty -> `True`
        #       3. `"node::johnson plc" in visited` -> `True` (added during Ex A's `_ingest_structured_tool_output`!)
        #     -> Enters `if` block below!
        if (
            tc_name in {"explore_graph_connections", "retrieve_location_info", "retrieve_supplier_info", "retrieve_product_info"}
            and granular_target
            and f"node::{granular_target}" in visited
        ):
            # Ex B (Call B2): Build a short-circuit JSON notice explaining that `"johnson plc"`
            # was already traversed by a multi-hop tool and is stored in `AgentState`:
            cached_summary = json.dumps(
                {
                    "redundant_1hop_call_blocked": True,
                    "entity": granular_target,
                    "message": (
                        f"Entity '{granular_target}' was ALREADY traversed by a multi-hop tool and "
                        "is stored in `AgentState` (`discovered_locations`, `discovered_suppliers`, "
                        "`discovered_products`). Do not make further 1-hop calls—synthesize your report now."
                    ),
                    "deterministic_summary_metrics": metrics_store,
                },
                indent=2,
            )
            # Ex B (Call B2): Record `signature` in `visited`, append `ToolMessage(tool_call_id="call_3")`, and `continue`:
            visited.add(signature)
            tool_messages.append(
                ToolMessage(content=cached_summary, name=tc_name, tool_call_id=tc_id)
            )
            continue

        # ---------------------------------------------------------------------
        # STEP 4: Execute the Tool Function (Reached by Example A; skipped in Example B)
        # ---------------------------------------------------------------------
        # Look up the Python function in `TOOL_MAP`:
        #   - Ex A: `tool_fn = analyze_location_blast_radius`
        tool_fn = TOOL_MAP.get(tc_name)
        if tool_fn is None:
            raw_result = json.dumps({"error": f"Unknown tool '{tc_name}'"})
        else:
            try:
                # Call `analyze_location_blast_radius(location_or_country_query="San Diego", risk_threshold=0.6)`:
                #   - Ex A `res`: JSON string `'{\n  "query": "San Diego",\n  "deterministic_summary_metrics": ...}'`
                res = tool_fn(**tc_args)
                # Ensure `raw_result` is a string (if a 1-hop tool returned a `list`, serialize it with `json.dumps`):
                #   - Ex A `raw_result`: `'{\n  "query": "San Diego", ...}'`
                raw_result = res if isinstance(res, str) else json.dumps(res)
            except Exception as exc:
                raw_result = json.dumps({"error": f"Tool '{tc_name}' failed: {exc}"})

        # Record the executed call signature in `visited`:
        #   - Ex A: adds `'analyze_location_blast_radius::{"location_or_country_query": "san diego", "risk_threshold": 0.6}'`
        visited.add(signature)
        if granular_target:
            visited.add(f"node::{granular_target}")

        # ---------------------------------------------------------------------
        # STEP 5: Ingest Structured Output into Cumulative Phase 2 `AgentState`
        # ---------------------------------------------------------------------
        # Parse `raw_result` and merge all discovered locations, suppliers, products,
        # node signatures, and deterministic metrics into our cumulative dictionaries:
        #   - Ex A mutations:
        #       `visited` adds `'node::san diego factory'`, `'node::johnson plc'`, `'node::grid monitoring meter 9971'`, ...
        #       `locations_by_name` adds `"San Diego Factory": {...}`
        #       `suppliers_by_name` adds `"Johnson PLC": {...}`, `"Rogers-Davis": {...}`
        #       `products_by_sku`   adds `"E-53774": {...}`, ...
        #       `metrics_store`     adds `{"total_locations_affected": 1, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74", ...}`
        _ingest_structured_tool_output(
            raw_result,
            visited=visited,
            locations_by_name=locations_by_name,
            suppliers_by_name=suppliers_by_name,
            products_by_sku=products_by_sku,
            metrics_store=metrics_store,
        )

        # Wrap `raw_result` in a LangChain `ToolMessage` linked to `tc_id` (`"call_1"`):
        #   - Ex A: `tool_messages = [ToolMessage(content='{"query": "San Diego", ...}', name="analyze_location_blast_radius", tool_call_id="call_1")]`
        tool_messages.append(
            ToolMessage(content=raw_result, name=tc_name, tool_call_id=tc_id)
        )

    # -------------------------------------------------------------------------
    # STEP 6: Return Updated Cumulative `AgentState` Fields to LangGraph
    # -------------------------------------------------------------------------
    # Note:
    #   - `"messages"` uses the `add_messages` reducer, so `tool_messages` are APPENDED to `state["messages"]`.
    #   - All other Phase 2 keys (`visited_nodes`, `discovered_locations`, `discovered_suppliers`,
    #     `discovered_products`, `deterministic_metrics`) contain the merged cumulative state
    #     from all turns so far (Step 2 rehydration + Step 5 ingestion).
    return {
        "messages": tool_messages,
        "visited_nodes": sorted(visited),
        "discovered_locations": list(locations_by_name.values()),
        "discovered_suppliers": list(suppliers_by_name.values()),
        "discovered_products": list(products_by_sku.values()),
        "deterministic_metrics": metrics_store,
        "reflection_feedback": state.get("reflection_feedback"),
        "reflection_count": state.get("reflection_count", 0),
    }


def _build_working_memory_digest(state: AgentState) -> str | None:
    """
    Build a compact, structured Working Memory digest from Phase 2 `AgentState`
    so that whichever model `openrouter/free` routes to on the current turn can
    immediately see all canonical entities and exact Python-computed totals without
    having to re-parse raw `ToolMessage` JSON strings.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Example A (Turn 1 — Before any tool has executed; `AgentState` only has `messages`):
      - Input `state`:
          `{"messages": [HumanMessage(content="Assess strike at San Diego")]}`
      - Output:
          `None`

    Example B (Turn 2 — After `stateful_tool_node` executed `analyze_location_blast_radius`):
      - Input `state`:
          ```python
          {
              "messages": [HumanMessage(...), AIMessage(...), ToolMessage(...)],
              "visited_nodes": ["analyze_location_blast_radius::{\"location_or_country_query\": \"san diego\", \"risk_threshold\": 0.6}", "node::san diego factory", ...],
              "discovered_locations": [{"location_name": "San Diego Factory", "country": "USA", "supplier_count": 2}],
              "discovered_suppliers": [{"name": "Johnson PLC", "location_name": "San Diego Factory", "country": "USA", "risk_score": 0.73, "risk_level": "HIGH_RISK", "revenue_formatted": "$124,049,854.50"}],
              "discovered_products": [{"name": "Grid Monitoring Meter 9971", "sku": "E-53774", "price_formatted": "$2,028.20", "supplier_name": "Johnson PLC", "location_name": "San Diego Factory"}],
              "deterministic_metrics": {"total_locations_affected": 1, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"},
              "reflection_feedback": None,
              "reflection_count": 0,
          }
          ```
      - Output:
          Multi-line string starting with `"=== PHASE 2 STRUCTURED WORKING MEMORY (CANONICAL GRAPH STATE) ==="`
          containing the formatted `[Deterministic Summary Metrics]`, `[Discovered Locations (1)]`,
          `[Discovered Suppliers (1)]`, and `[Discovered Products (1)]` sections.
    ===========================================================================
    """
    # Unpack canonical entity lists and deterministic metrics from `state` (falling back to empty list/dict):
    #   - Ex A (Turn 1): `locs = []`, `sups = []`, `prods = []`, `metrics = {}`
    #   - Ex B (Turn 2):
    #       `locs    = [{"location_name": "San Diego Factory", "country": "USA", ...}]`
    #       `sups    = [{"name": "Johnson PLC", "risk_score": 0.73, "revenue_formatted": "$124,049,854.50", ...}]`
    #       `prods   = [{"name": "Grid Monitoring Meter 9971", "sku": "E-53774", "price_formatted": "$2,028.20", ...}]`
    #       `metrics = {"total_locations_affected": 1, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"}`
    locs = state.get("discovered_locations") or []
    sups = state.get("discovered_suppliers") or []
    prods = state.get("discovered_products") or []
    metrics = state.get("deterministic_metrics") or {}

    # If all 4 containers are empty (Turn 1 before any tool runs), return `None` so `chatbot`
    # uses the base system prompt without an empty Working Memory block:
    #   - Ex A: `not ([] or [] or [] or {})` -> `True` -> returns `None` immediately!
    #   - Ex B: `not ([...] or [...] or [...] or {...})` -> `False` -> continues to build digest
    if not (locs or sups or prods or metrics):
        return None

    # Initialize `lines` with the Phase 2 Working Memory header instructions:
    #   - Ex B `lines`: 3 header strings instructing the LLM to use these exact entities/metrics
    lines = [
        "=== PHASE 2 STRUCTURED WORKING MEMORY (CANONICAL GRAPH STATE) ===",
        "Use these exact Python-verified metrics and canonical entities in your response.",
        "Do NOT call additional tools for entities already listed below:",
    ]

    # Format `[Deterministic Summary Metrics]` section if `metrics` is non-empty:
    #   - Ex B appends:
    #       "\n[Deterministic Summary Metrics (Cumulative Across All Discovered Entities)]"
    #       "  - total_locations_affected: 1"
    #       "  - total_supplier_revenue_exposed_formatted: $1,113,139,177.74"
    #     (If 2+ queries ran, e.g. "San Diego" and "Rotterdam", also prints `[Per-Query Metrics Breakdown]`.)
    if metrics:
        lines.append("\n[Deterministic Summary Metrics (Cumulative Across All Discovered Entities)]")
        for k, v in metrics.items():
            if k == "per_query_breakdown":
                continue
            lines.append(f"  - {k}: {v}")
        per_query = metrics.get("per_query_breakdown")
        if isinstance(per_query, dict) and len(per_query) > 1:
            lines.append("\n[Per-Query Metrics Breakdown]")
            for q_label, q_metrics in per_query.items():
                lines.append(f"  * Query '{q_label}':")
                for mk, mv in q_metrics.items():
                    lines.append(f"      - {mk}: {mv}")

    # Format `[Discovered Locations (N)]` section if `locs` is non-empty:
    #   - Ex B appends:
    #       "\n[Discovered Locations (1)]"
    #       "  - San Diego Factory (USA)"
    if locs:
        lines.append(f"\n[Discovered Locations ({len(locs)})]")
        for loc in locs:
            lines.append(f"  - {loc.get('location_name')} ({loc.get('country')})")

    # Format `[Discovered Suppliers (N)]` section if `sups` is non-empty:
    #   - Ex B appends:
    #       "\n[Discovered Suppliers (1)]"
    #       "  - Johnson PLC @ San Diego Factory, USA | risk_score=0.73 (HIGH_RISK) | revenue=$124,049,854.50"
    if sups:
        lines.append(f"\n[Discovered Suppliers ({len(sups)})]")
        for sup in sups:
            lines.append(
                f"  - {sup.get('name')} @ {sup.get('location_name')}, {sup.get('country')} | "
                f"risk_score={sup.get('risk_score')} ({sup.get('risk_level')}) | "
                f"revenue={sup.get('revenue_formatted')}"
            )

    # Format `[Discovered Products (N)]` section if `prods` is non-empty:
    #   - Ex B appends:
    #       "\n[Discovered Products (1)]"
    #       "  - Grid Monitoring Meter 9971 (SKU: E-53774, $2,028.20) <- Supplier: Johnson PLC (San Diego Factory)"
    if prods:
        lines.append(f"\n[Discovered Products ({len(prods)})]")
        for prod in prods:
            lines.append(
                f"  - {prod.get('name')} (SKU: {prod.get('sku')}, {prod.get('price_formatted')}) "
                f"<- Supplier: {prod.get('supplier_name')} ({prod.get('location_name')})"
            )

    # Append optional self-reflection critique if set (used in Phase 3 CRAG):
    #   - Ex B: `feedback = None` -> skipped
    feedback = state.get("reflection_feedback")
    if feedback:
        lines.append(f"\n[Verification Feedback to Address]\n  {feedback}")

    # Join all lines with newline characters into a single compact string:
    #   - Ex B Output: `"=== PHASE 2 STRUCTURED WORKING MEMORY ... <- Supplier: Johnson PLC (San Diego Factory)"`
    return "\n".join(lines)


api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is required when using OpenRouter. "
        "Get your free key from https://openrouter.ai/"
    )

model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model=model_name,
    temperature=0,
    max_retries=5,
)

llm_with_tools = llm.bind_tools(tools)

PHASE1_SYSTEM_PROMPT = (
    "You are an elite Global Supply Chain Risk Analysis Agent powered by GraphRAG.\n"
    "RULES FOR ROBUST & ACCURATE ANALYSIS:\n"
    "1. For ANY location, city, port, or country disruption query, call `analyze_location_blast_radius` FIRST. "
    "It automatically resolves typos, traverses the full 2-hop subgraph (Location -> Supplier -> Product), "
    "and calculates exact financial exposure in Python.\n"
    "2. For ANY product or category risk audit (e.g. Pharmaceutical, medical, electronics), call "
    "`audit_products_and_upstream_suppliers` FIRST. It finds all matching products and maps their upstream "
    "suppliers and locations in a single call.\n"
    "3. Once a composite tool (`analyze_location_blast_radius` or `audit_products_and_upstream_suppliers`) "
    "populates the Phase 2 Structured Working Memory (`discovered_locations`, `discovered_suppliers`, "
    "`discovered_products`, and `deterministic_metrics`), DO NOT make redundant calls to "
    "`explore_graph_connections` or `retrieve_*` for those same entities—immediately write your final report.\n"
    "4. NEVER perform mental math for revenue totals or counts. Always quote the exact dollar figures "
    "from `deterministic_summary_metrics` verbatim, and list EVERY affected location, supplier, and product "
    "returned by the tool without omitting any items."
)


def chatbot(state: AgentState):
    """
    Phase 1+2 LangGraph Reasoning Node (`chatbot`).

    Reads the current `AgentState`, refreshes the leading `SystemMessage` with the
    latest Phase 2 Structured Working Memory digest, enforces tool-call loop limits,
    and invokes `llm_with_tools`.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Example A (Turn 1 — Initial user query enters the graph):
      - Input `state`:
          `{"messages": [HumanMessage(content="Assess the impact of a strike at San Diego")]}`
      - Execution Flow:
          1. `total_tool_invocations = 0`, `ai_tool_turns = 0`
          2. `_build_working_memory_digest(state)` -> `None`
          3. Prepends `SystemMessage(content=PHASE1_SYSTEM_PROMPT)` before `HumanMessage`
          4. `llm_with_tools.invoke(messages)` -> returns `AIMessage` with 1 tool call:
             `analyze_location_blast_radius(location_or_country_query="San Diego", risk_threshold=0.6)`
      - Output:
          `{"messages": [AIMessage(tool_calls=[...])], "visited_nodes": [], "discovered_locations": [], ...}`
          (LangGraph `tools_condition` sees `tool_calls` on the `AIMessage` and routes to `stateful_tool_node`.)

    Example B (Turn 2 — After `stateful_tool_node` executed the tool and populated `AgentState`):
      - Input `state`:
          ```python
          {
              "messages": [
                  HumanMessage(content="Assess the impact of a strike at San Diego"),
                  AIMessage(content="", tool_calls=[{"id": "call_1", "name": "analyze_location_blast_radius", "args": {"location_or_country_query": "San Diego", "risk_threshold": 0.6}}]),
                  ToolMessage(content='{"deterministic_summary_metrics": ...}', tool_call_id="call_1"),
              ],
              "visited_nodes": ["analyze_location_blast_radius::{\"location_or_country_query\": \"san diego\", \"risk_threshold\": 0.6}", "node::san diego factory", ...],
              "discovered_locations": [{"location_name": "San Diego Factory", "country": "USA", ...}],
              "discovered_suppliers": [{"name": "Johnson PLC", "risk_score": 0.73, ...}],
              "discovered_products": [{"name": "Grid Monitoring Meter 9971", "sku": "E-53774", ...}],
              "deterministic_metrics": {"total_locations_affected": 1, "total_supplier_revenue_exposed_formatted": "$1,113,139,177.74"},
          }
          ```
      - Execution Flow:
          1. `total_tool_invocations = 1`, `ai_tool_turns = 1`
          2. `_build_working_memory_digest(state)` -> returns the formatted Phase 2 Working Memory block
          3. Combines `PHASE1_SYSTEM_PROMPT + "\n\n" + working_memory_digest` into `SystemMessage`
          4. `llm_with_tools.invoke(messages)` -> sees the complete subgraph + Working Memory and generates
             the final Markdown risk report (`AIMessage(content="## 🚨 SAN DIEGO STRIKE IMPACT...", tool_calls=[])`)
      - Output:
          `{"messages": [AIMessage(content="## 🚨 SAN DIEGO STRIKE IMPACT...")], "visited_nodes": [...], "discovered_locations": [...], ...}`
          (LangGraph `tools_condition` sees no `tool_calls` and routes to `__end__`.)
    ===========================================================================
    """
    # Copy `state["messages"]` into a local list so we can prepend/update `SystemMessage`
    # without mutating `state` in place:
    #   - Ex A (Turn 1) `messages`: `[HumanMessage("Assess the impact of a strike at San Diego")]` (len = 1)
    #   - Ex B (Turn 2) `messages`: `[HumanMessage(...), AIMessage(tool_calls=[...]), ToolMessage(...)]` (len = 3)
    messages = list(state.get("messages", []))

    # Count total individual tool invocations across all `AIMessage`s so far
    # (fixes the baseline `agent.py` bug where a single turn with 5 parallel tool calls was counted as 1):
    #   - Ex A (Turn 1): `sum([0])` -> `total_tool_invocations = 0`
    #   - Ex B (Turn 2): `sum([0, 1, 0])` -> `total_tool_invocations = 1`
    total_tool_invocations = sum(
        len(getattr(msg, "tool_calls", []) or []) for msg in messages
    )

    # Count how many AI turns have dispatched tool calls so far:
    #   - Ex A (Turn 1): `ai_tool_turns = 0`
    #   - Ex B (Turn 2): `ai_tool_turns = 1`
    ai_tool_turns = sum(
        1 for msg in messages if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    # Build the Phase 2 Working Memory digest from `state`:
    #   - Ex A (Turn 1): `working_memory_digest = None` (no tools have run yet)
    #   - Ex B (Turn 2): `working_memory_digest = "=== PHASE 2 STRUCTURED WORKING MEMORY ..."`
    working_memory_digest = _build_working_memory_digest(state)

    # Start with `PHASE1_SYSTEM_PROMPT` and append `working_memory_digest` if present:
    #   - Ex A (Turn 1): `system_content = PHASE1_SYSTEM_PROMPT`
    #   - Ex B (Turn 2): `system_content = PHASE1_SYSTEM_PROMPT + "\n\n=== PHASE 2 STRUCTURED WORKING MEMORY ..."`
    system_content = PHASE1_SYSTEM_PROMPT
    if working_memory_digest:
        system_content = f"{PHASE1_SYSTEM_PROMPT}\n\n{working_memory_digest}"

    # Place `SystemMessage(content=system_content)` at index 0 of `messages`:
    #   - Ex A (Turn 1): `messages[0]` is `HumanMessage`, so `else` branch prepends `SystemMessage`
    #     -> `messages = [SystemMessage(PHASE1_SYSTEM_PROMPT), HumanMessage(...)]`
    #   - Ex B (Turn 2): `messages[0]` is `HumanMessage` (since `SystemMessage` is not persisted into `state["messages"]`),
    #     so `else` branch prepends the upgraded `SystemMessage(PHASE1_SYSTEM_PROMPT + Working Memory)`
    #     -> `messages = [SystemMessage(PHASE1_SYSTEM_PROMPT + Working Memory), HumanMessage(...), AIMessage(...), ToolMessage(...)]`
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_content)
    else:
        messages = [SystemMessage(content=system_content)] + messages

    # Safety valve: If the model has already taken >= 8 tool turns or >= 16 total tool calls,
    # append an explicit stop reminder so it synthesizes the final answer immediately:
    #   - Ex A (Turn 1): `0 >= 8 or 0 >= 16` is `False` (skipped)
    #   - Ex B (Turn 2): `1 >= 8 or 1 >= 16` is `False` (skipped)
    if ai_tool_turns >= 8 or total_tool_invocations >= 16:
        reminder = SystemMessage(
            content="IMPORTANT: You already have the complete subgraph and deterministic metrics "
                    "in the Phase 2 Working Memory above. Do NOT call any more tools. Synthesize your final "
                    "comprehensive risk assessment report now, quoting the exact Python-computed totals."
        )
        messages = messages + [reminder]

    # Call the OpenRouter LLM with bound tools:
    #   - Ex A (Turn 1) Output `response`:
    #       `AIMessage(content="", tool_calls=[{"name": "analyze_location_blast_radius", "args": {"location_or_country_query": "San Diego", "risk_threshold": 0.6}, "id": "call_1"}])`
    #   - Ex B (Turn 2) Output `response`:
    #       `AIMessage(content="## 🚨 SAN DIEGO STRIKE IMPACT ASSESSMENT\n...", tool_calls=[])`
    response = llm_with_tools.invoke(messages)

    # Return the new `AIMessage` (appended by `add_messages` reducer) while preserving all
    # Phase 2 structured `AgentState` fields across turns:
    #   - Ex A Output dict: `{"messages": [AIMessage(tool_calls=[...])], "visited_nodes": [], "discovered_locations": [], ...}`
    #   - Ex B Output dict: `{"messages": [AIMessage(content="## 🚨 SAN DIEGO...")], "visited_nodes": [...], "discovered_locations": [...], ...}`
    return {
        "messages": [response],
        "visited_nodes": state.get("visited_nodes", []),
        "discovered_locations": state.get("discovered_locations", []),
        "discovered_suppliers": state.get("discovered_suppliers", []),
        "discovered_products": state.get("discovered_products", []),
        "deterministic_metrics": state.get("deterministic_metrics", {}),
        "reflection_feedback": state.get("reflection_feedback"),
        "reflection_count": state.get("reflection_count", 0),
    }


# =====================================================================
# PHASE 3: CORRECTIVE RAG (CRAG) / DETERMINISTIC VERIFICATION & REFLECTION
# =====================================================================

MAX_REFLECTION_RETRIES = 2


def verify_and_reflect(state: AgentState) -> dict:
    """
    Phase 3 Deterministic Fact, Completeness, and Math Guardrail Node (`verify_and_reflect`).

    Runs immediately after `chatbot` generates a candidate final answer (with no `tool_calls`)
    and performs two fast (<1 ms) deterministic checks against Phase 2 `AgentState`:
      1. Completeness Check:
         - Verifies every discovered `Location` (`location_name`) is mentioned in the final report.
         - Verifies every discovered `Supplier` (`name`) is mentioned in the final report.
         - Verifies every discovered `Product` (`name` OR `sku`) is mentioned in the final report.
      2. Numerical Grounding Check:
         - Verifies that the exact Python-computed `total_supplier_revenue_exposed_formatted`
           (e.g. `"$1,113,139,177.74"`) appears in the final report so no mental-math errors slip through.

    ===========================================================================
    RUNNING TEST CASES USED IN LINE-BY-LINE COMMENTS BELOW
    ===========================================================================
    Shared Phase 2 Working Memory in `state` for both examples:
      - `discovered_locations = [{"location_name": "San Diego Factory", "country": "USA"}]`
      - `discovered_suppliers = [{"name": "Johnson PLC", "risk_score": 0.73}, {"name": "Rogers-Davis", "risk_score": 0.14}]`
      - `discovered_products  = [{"name": "Grid Monitoring Meter 9971", "sku": "E-53774"}, {"name": "Pallet Wrap Film 5478", "sku": "P-99143"}]`
      - `deterministic_metrics = {"total_supplier_revenue_exposed_formatted": "$1,113,139,177.74", "total_supplier_revenue_exposed": 1113139177.74}`

    Example A (Complete & Numerically Grounded Report -> PASSES on first check):
      - Last `AIMessage.content`:
          `"San Diego Factory impact: Johnson PLC (Grid Monitoring Meter 9971 / E-53774) and Rogers-Davis (Pallet Wrap Film 5478 / P-99143). Total Revenue Exposed: $1,113,139,177.74."`
      - Execution Flow:
          - `missing_locations = []`, `missing_suppliers = []`, `missing_products = []`, `math_issues = []`
          - `issues = []` (All checks pass!)
      - Output:
          `{"reflection_feedback": None, "reflection_count": 0}`
          (`route_after_verify` sees `reflection_feedback is None` and routes to `"__end__"`.)

    Example B (Incomplete Report + Mental-Math Error -> FAILS & triggers self-correction):
      - Last `AIMessage.content`:
          `"San Diego Factory impact: Johnson PLC makes Grid Monitoring Meter 9971 (E-53774). Total Revenue Exposed: $900,000,000.00."`
          (Notice the LLM omitted supplier `"Rogers-Davis"`, omitted product `"Pallet Wrap Film 5478 (SKU: P-99143)"`,
           and hallucinated `"$900,000,000.00"` instead of `"$1,113,139,177.74"`!)
      - Execution Flow:
          - `missing_suppliers = ["Rogers-Davis"]`
          - `missing_products  = ["Pallet Wrap Film 5478 (SKU: P-99143)"]`
          - `math_issues       = ["Exact total revenue exposed '$1,113,139,177.74' is missing from the report."]`
          - Since `reflection_count` (`0`) < `MAX_REFLECTION_RETRIES` (`2`), constructs `feedback_msg` (`HumanMessage`)
            listing the exact missing items and exact dollar figure, and increments `reflection_count` to `1`.
      - Output:
          `{"messages": [HumanMessage(content="⚠️ [PHASE 3 SELF-CORRECTION REQUIRED]...")], "reflection_feedback": "...", "reflection_count": 1}`
          (`route_after_verify` sees `reflection_feedback` is non-empty and routes back to `"chatbot"`!)
    ===========================================================================
    """
    # Extract conversation history and current `reflection_count` from `state`:
    #   - Ex A & Ex B: `reflection_count = 0`
    messages = state.get("messages", [])
    reflection_count = int(state.get("reflection_count") or 0)

    # Grab the candidate final answer text from the last `AIMessage`:
    #   - Ex A `answer_text`: `"San Diego Factory impact: Johnson PLC ... $1,113,139,177.74."`
    #   - Ex B `answer_text`: `"San Diego Factory impact: Johnson PLC ... $900,000,000.00."`
    last_msg = messages[-1] if messages else None
    answer_text = str(getattr(last_msg, "content", "") or "")
    answer_lower = answer_text.lower()

    # Unpack canonical entities and deterministic metrics from Phase 2 `AgentState`:
    locs = state.get("discovered_locations") or []
    sups = state.get("discovered_suppliers") or []
    prods = state.get("discovered_products") or []
    metrics = state.get("deterministic_metrics") or {}

    # If no entities were discovered (e.g., query matched 0 locations in the graph), pass immediately:
    if not (locs or sups or prods):
        return {
            "reflection_feedback": None,
            "reflection_count": reflection_count,
        }

    # -------------------------------------------------------------------------
    # CHECK 1: Entity Completeness Verification (Locations, Suppliers, Products)
    # -------------------------------------------------------------------------
    # 1a. Check that every discovered `location_name` appears in `answer_lower`:
    #   - Ex A & Ex B: `"san diego factory" in answer_lower` -> `True` -> `missing_locations = []`
    missing_locations = [
        loc["location_name"]
        for loc in locs
        if loc.get("location_name") and loc["location_name"].lower() not in answer_lower
    ]

    # 1b. Check that every discovered supplier `name` appears in `answer_lower`:
    #   - Ex A: both `"johnson plc"` and `"rogers-davis"` are in `answer_lower` -> `missing_suppliers = []`
    #   - Ex B: `"rogers-davis" not in answer_lower` -> `missing_suppliers = ["Rogers-Davis"]`
    missing_suppliers = [
        sup["name"]
        for sup in sups
        if sup.get("name") and sup["name"].lower() not in answer_lower
    ]

    # 1c. Check that every discovered product is mentioned by either its `name` OR its `sku`:
    #   - Ex A: both `"e-53774"` and `"p-99143"` are in `answer_lower` -> `missing_products = []`
    #   - Ex B: neither `"pallet wrap film 5478"` nor `"p-99143"` is in `answer_lower`
    #           -> `missing_products = ["Pallet Wrap Film 5478 (SKU: P-99143)"]`
    missing_products = []
    for prod in prods:
        p_name = (prod.get("name") or "").strip()
        p_sku = (prod.get("sku") or "").strip()
        name_present = bool(p_name and p_name.lower() in answer_lower)
        sku_present = bool(p_sku and p_sku.lower() in answer_lower)
        if not (name_present or sku_present):
            missing_products.append(f"{p_name} (SKU: {p_sku})")

    # -------------------------------------------------------------------------
    # CHECK 2: Deterministic Numerical Grounding Verification
    # -------------------------------------------------------------------------
    # Verify that the exact Python-computed `total_supplier_revenue_exposed_formatted`
    # (either with commas `"$1,113,139,177.74"` or raw `"1113139177.74"`) is present:
    #   - Ex A: `"$1,113,139,177.74" in answer_text` -> `True` -> `math_issues = []`
    #   - Ex B: `"$1,113,139,177.74" in answer_text` -> `False` -> `math_issues = ["Exact total revenue exposed '$1,113,139,177.74' is missing from the report."]`
    math_issues: list[str] = []
    expected_total_fmt = metrics.get("total_supplier_revenue_exposed_formatted")
    expected_total_raw = metrics.get("total_supplier_revenue_exposed")
    if expected_total_fmt and expected_total_raw:
        # Strip "$" for flexible check (e.g. "1,113,139,177.74" or "1113139177.74")
        clean_fmt = str(expected_total_fmt).replace("$", "").strip()
        clean_raw = f"{float(expected_total_raw):.2f}"
        if clean_fmt not in answer_text and clean_raw not in answer_text:
            math_issues.append(
                f"Exact total supplier revenue exposed ({expected_total_fmt}) is missing or miscalculated in your report."
            )

    # -------------------------------------------------------------------------
    # Combine all detected issues and decide whether to PASS or REFLECT
    # -------------------------------------------------------------------------
    issues: list[str] = []
    if missing_locations:
        issues.append(f"Missing Location(s) ({len(missing_locations)}): {', '.join(missing_locations)}")
    if missing_suppliers:
        issues.append(f"Missing Supplier(s) ({len(missing_suppliers)}): {', '.join(missing_suppliers)}")
    if missing_products:
        issues.append(f"Missing Product(s) ({len(missing_products)}): {', '.join(missing_products)}")
    if math_issues:
        issues.extend(math_issues)

    # Case 1: All checks passed (or we already reached `MAX_REFLECTION_RETRIES`):
    #   - Ex A: `issues` is `[]` -> returns `{"reflection_feedback": None, "reflection_count": 0}`
    if not issues or reflection_count >= MAX_REFLECTION_RETRIES:
        return {
            "reflection_feedback": None,
            "reflection_count": reflection_count,
        }

    # Case 2: Verification failed and `reflection_count < MAX_REFLECTION_RETRIES`:
    #   - Ex B: Builds explicit critique and appends a `HumanMessage` instructing `chatbot`
    #     to regenerate the complete report using the Phase 2 Working Memory without calling tools.
    feedback_text = " | ".join(issues)
    correction_prompt = (
        "⚠️ [PHASE 3 DETERMINISTIC VERIFICATION FAILED — SELF-CORRECTION REQUIRED]\n"
        "Your draft report failed our automated completeness/math audit for the following reason(s):\n"
        + "\n".join(f"  - {issue}" for issue in issues)
        + "\n\nPlease rewrite the complete final report now. Include EVERY missing entity listed above "
        "and quote the exact dollar figures from `[Deterministic Summary Metrics]` in Working Memory. "
        "Do NOT call any tools—all data is already in your Phase 2 Working Memory."
    )

    return {
        "messages": [HumanMessage(content=correction_prompt)],
        "reflection_feedback": feedback_text,
        "reflection_count": reflection_count + 1,
    }


def route_after_chatbot(state: AgentState) -> str:
    """
    Conditional edge router after `chatbot`:
      - If the last `AIMessage` has `tool_calls`, route to `"tools"` (`stateful_tool_node`).
      - Otherwise (candidate final report), route to `"verify_and_reflect"` (Phase 3 CRAG node).
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "verify_and_reflect"


def route_after_verify(state: AgentState) -> str:
    """
    Conditional edge router after `verify_and_reflect`:
      - If `reflection_feedback` is set (verification failed and retries remain), route back to `"chatbot"`.
      - Otherwise (verification passed!), route to `"__end__"`.
    """
    if state.get("reflection_feedback"):
        return "chatbot"
    return "__end__"


builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", stateful_tool_node)
builder.add_node("verify_and_reflect", verify_and_reflect)

builder.add_edge("__start__", "chatbot")
builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
    {"tools": "tools", "verify_and_reflect": "verify_and_reflect"},
)
builder.add_edge("tools", "chatbot")
builder.add_conditional_edges(
    "verify_and_reflect",
    route_after_verify,
    {"chatbot": "chatbot", "__end__": "__end__"},
)

agent = builder.compile()

