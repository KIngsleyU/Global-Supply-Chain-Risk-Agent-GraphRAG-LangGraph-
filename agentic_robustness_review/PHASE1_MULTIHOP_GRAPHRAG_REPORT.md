# Phase 1 Architecture & Multi-Hop GraphRAG Report

> [!NOTE]
> **What is Phase 1?**
> Phase 1 upgrades the agent's **tool and retrieval layer** inside [`agent_phase1.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py) and [`main_phase1.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/main_phase1.py) (while leaving the root Phase 0 [`agent.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agent.py) and [`main.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/main.py) untouched). It replaces Phase 0's $1 + L + S$ sequential 1-hop lookups (`nx.all_neighbors`), hardcoded `k=5` vector truncation, and error-prone LLM mental arithmetic with **2-hop composite GraphRAG traversal tools**, **3-tier typo-tolerant location resolution**, **external alternative-supplier discovery**, and **exact Python financial aggregation**.

---

## 1. Phase 1 Agent Architecture Diagram

The diagram below shows how Phase 1 collapses 15–20 sequential tool calls into a **single composite GraphRAG tool execution** ($<1\text{ ms}$ in NetworkX) and returns both the full 2-hop subgraph and exact Python-computed financial metrics in one JSON payload.

```mermaid
flowchart TD
    subgraph Entry["1. CLI & Visualization Entry (main_phase1.py & app.py)"]
        UserQuery(["User Risk Query\nmain_phase1.py -> run_phase1_agent_with_live_visibility()"])
        RiskUI["Streamlit Risk Map UI (app.py)\nlaunch_risk_map_ui() -> SC_GRAPH_DOT_PATH\n(stdout/stderr silenced; no duplicate graph gen)"]
        UserQuery -. "Spawns subprocess" .-> RiskUI
    end

    subgraph LangGraph["2. Phase 1 LangGraph Execution Loop (agent_phase1.py)"]
        StartNode(["__start__"])
        Chatbot["chatbot(state) Node\n• Prepends PHASE1_SYSTEM_PROMPT\n• Tracks total_tool_invocations (>=16) & ai_tool_turns (>=8)\n• Invokes ChatOpenAI (openrouter/free)"]
        Router{"Tool Call Router\nHas tool_calls?"}
        ToolExec["Tool Execution Node\nDispatches composite 2-hop tool\n(1 call replaces 15-20 1-hop calls)"]
        EndNode(["Final Markdown Risk Report\nQuotes exact Python metrics & full subgraph"])

        StartNode --> Chatbot --> Router
        Router -- "Yes (Turn 1)" --> ToolExec --> Chatbot
        Router -- "No (Turn 2)" --> EndNode
    end

    subgraph Phase1Tools["3. Phase 1 Composite 2-Hop GraphRAG Tools (agent_phase1.py)"]
        subgraph BlastTool["A. Downstream Location Tool: analyze_location_blast_radius()"]
            LocRes["_resolve_locations_hybrid(query)\n• Step 0: Strip LOCATION_STOPWORDS\n• Step 1: Exact word-boundary regex (\\b...\\b)\n• Step 2: 4+ char token match\n• Step 3: Fuzzy SequenceMatcher (>=0.72)"]
            HopDown["2-Hop Directed Traversal (<1 ms)\nLocation <--LOCATED_AT-- Supplier --MANUFACTURES--> Product"]
            AltScan["External Backup Supplier Scan\nFinds non-disrupted suppliers making\nsame base product type + flags sole-source lines"]
            PyMathA["Deterministic Python Rollup\n• total_supplier_revenue_exposed\n• high_risk_supplier_revenue_exposed\n• risk_weighted_revenue_exposure"]
            LocRes --> HopDown --> AltScan --> PyMathA
        end

        subgraph AuditTool["B. Upstream Product Tool: audit_products_and_upstream_suppliers()"]
            ProdRes["3-Tier Product Resolution (No k=5 Ceiling)\n• Tier 1: CATEGORY_KEYWORDS catalog match\n• Tier 2: Direct name/SKU substring match\n• Guard: Block misrouted location queries\n• Tier 3: ChromaDB semantic fallback (k=25)"]
            HopUp["2-Hop Upstream Traversal (<1 ms)\nProduct <--MANUFACTURES-- Supplier --LOCATED_AT--> Location"]
            PyMathB["Deterministic Python Rollup\nSorts by risk_score desc & sums\ntotal + high-risk supplier revenue"]
            ProdRes --> HopUp --> PyMathB
        end

        FallbackTools["C. Retained Fallback Tools (Raised Ceilings)\n• retrieve_product_info (k=15)\n• retrieve_supplier_info (k=10)\n• retrieve_location_info (k=10)\n• explore_graph_connections (1-hop)"]
    end

    UserQuery --> StartNode
    ToolExec --> BlastTool & AuditTool & FallbackTools
```

---

## 2. The Story of Phase 1: Step-by-Step Code Walkthrough

### Step 1: Word-Boundary & Typo-Resilient Location Resolution ([`_resolve_locations_hybrid`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L139-L276))
In Phase 0, [`get_node_by_name`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/graph_ops.py#L80-L90) failed on country queries (`"China"`), partial names (`"San Diego"`), or typos (`"Chna"`, `"shanghia"`), while naive substring matching caused 3-letter `"san"` in `"San Diego"` to falsely match `"Port of Busan"`. [`_resolve_locations_hybrid`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L139-L276) solves both problems using a 4-stage pipeline:
- **Step 0 — Stopword Stripping ([`lines 176–188`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L176-L188))**: Strips [`LOCATION_STOPWORDS`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L131-L136) (`"port"`, `"warehouse"`, `"factory"`, `"in"`, `"of"`, etc.) so `"Port of San Diego"` becomes `clean_geo_query = "san diego"`.
- **Step 1 — Exact Word-Boundary Match ([`lines 196–211`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L196-L211))**: Tests regex `\b{clean_geo_query}\b` against `loc.name.lower()` and `loc.country.lower()`. Matches `"San Diego Factory"` (`"direct_match"`) while never matching `"Busan"`.
- **Step 2 — Multi-City Token Match ([`lines 219–234`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L219-L234))**: Filters to tokens with `len(w) >= 4` and checks exact word membership in `loc_words` (`"token_match"`).
- **Step 3 — Fuzzy Edit-Distance Match ([`lines 243–273`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L243-L273))**: Uses `difflib.SequenceMatcher(None, q_tok, target).ratio() >= 0.72` to resolve country typos (`"chna"` $\rightarrow$ `"china"`, ratio `0.89`, returning all Chinese ports) and city typos (`"shanghia"` $\rightarrow$ `"shanghai"`, ratio `0.88`).

### Step 2: Downstream 2-Hop Traversal, Backup Suppliers, & Exact Math ([`analyze_location_blast_radius`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L283-L550))
Instead of forcing the LLM to make $1 + L + S$ separate tool calls, [`analyze_location_blast_radius`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L283-L550) executes the entire downstream impact analysis in Python:
1. **2-Hop Graph Traversal ([`lines 363–421`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L363-L421))**: For every resolved `Location`, traverses Hop 1 (`connected_suppliers` via `isinstance(node, Supplier)`) and Hop 2 (`connected_products` via `isinstance(node, Product)`), deduplicating entities in `unique_suppliers` and `unique_products`.
2. **Deterministic Financial Aggregation ([`lines 428–441`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L428-L441))**: Computes exact Python floats and formatted dollar strings for:
   - `total_revenue = round(sum(s.revenue for s in unique_suppliers.values()), 2)`
   - `high_risk_revenue = round(sum(s.revenue for s in high_risk_sups), 2)` (where `s.risk_score > risk_threshold`)
   - `risk_weighted_revenue = round(sum(s.revenue * s.risk_score for s in unique_suppliers.values()), 2)`
3. **External Alternative Supplier Discovery ([`lines 447–521`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L447-L521))**: Strips trailing model numbers from affected products (e.g., `"Industrial Router 101"` $\rightarrow$ base type `"Industrial Router"`), scans all non-disrupted locations (`other_loc not in matched_loc_set`), and identifies every external backup supplier (`alternative_suppliers_by_product_type`), its current revenue, and any sole-source product lines with zero external backups (`unmatched_product_types`).

### Step 3: Upstream Product-to-Supplier Audit Without `k=5` Truncation ([`audit_products_and_upstream_suppliers`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L553-L700))
To fix Run 3's silent `k=5` truncation (where 37.5% of pharmaceutical products were missed), [`audit_products_and_upstream_suppliers`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L553-L700) resolves products in 3 tiers before traversing 2 hops upstream:
1. **Catalog & Keyword Matching ([`lines 580–598`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L580-L598))**: Matches category names (`"pharmaceutical"`, `"medical"`, `"electronics"`, etc.) against [`CATEGORY_KEYWORDS`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L99-L128) and direct product/SKU substrings so **100% of category products** are retrieved regardless of count.
2. **Misrouted Location Guard & `k=25` Semantic Fallback ([`lines 603–625`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L603-L625))**: Blocks accidental geographic queries (`"misrouted_location_query_blocked": True`) and raises the ChromaDB fallback ceiling from `k=5` to `k=25`.
3. **2-Hop Upstream Join & Risk Sorting ([`lines 631–700`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L631-L700))**: Traverses `Product -> Supplier -> Location`, sorts results by `supplier_risk_score` descending, and computes exact revenue totals.

### Step 4: Accurate Tool-Call Accounting & Rich Subgraph Terminal Visibility ([`chatbot`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L1635-L1762) & [`main_phase1.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/main_phase1.py#L76-L156))
- **Fixed Loop Counter ([`agent_phase1.py#L1696-L1740`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L1696-L1740))**: Counts both individual tool invocations (`total_tool_invocations = sum(len(msg.tool_calls))`) and AI tool turns (`ai_tool_turns`), pairing them with [`PHASE1_SYSTEM_PROMPT`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py#L1616-L1632) rules that forbid mental math and redundant 1-hop calls.
- **Full Subgraph Terminal Formatting ([`format_tool_output` in `main_phase1.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/main_phase1.py#L76-L156))**: Parses the composite tool JSON and prints `[Deterministic Summary Metrics]`, `[Traversed Subgraph: Location -> Supplier -> Product]`, `[Alternative Suppliers Outside Disrupted Zone]`, and `[Traversed Subgraph: Product -> Supplier -> Location]` directly in the console trace so every entity used by the LLM is transparently visible.

---

## 3. Empirical Impact: Phase 0 Baseline vs. Phase 1 Benchmark

Evaluated across the 3 benchmark scenarios in [`agent_eval.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_eval.py#L169-L227) (`python agentic_robustness_review/agent_eval.py --phase1`, saved in [`eval_results_phase1.json`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/eval_results_phase1.json)):

| Metric | Phase 0 Baseline ([`agent.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agent.py)) | Phase 1 Upgraded ([`agent_phase1.py`](file:///Applications/School/ventures/Global-Supply-Chain-Risk-Agent-GraphRAG-LangGraph-/agentic_robustness_review/agent_phase1.py)) | Phase 1 Improvement |
| :--- | :--- | :--- | :--- |
| **Run 1 Tool Calls / Messages** | `15` calls / `22` msgs (`25.90s`) | **`1` call / `4` msgs (`12.25s`)** | **-93% tool calls, 2.1x faster** |
| **Run 2 Tool Calls / Messages** | `19` calls / `27` msgs (`34.22s`) | **`1` call / `4` msgs (`31.98s`)** | **-95% tool calls, 0 wasted turns** |
| **Run 2 Product Recall** | `81.25%` (`13/16` reported) | **`100%` (`10/10` reported)** | **+18.75% recall (0 dropped items)** |
| **Run 2 Revenue Math Error** | Off by `$300.4M` (`7.6%` error) | **`$0.00` error (`$2,831,125,219.15` exact)** | **100% deterministic math accuracy** |
| **Run 3 Tool Calls / Messages** | `11` calls / `17` msgs (`22.53s`) | **`1` call / `4` msgs (`22.80s`)** | **-91% tool calls** |
| **Run 3 Pharma Product Recall** | `62.5%` (`5/8`, truncated by `k=5`) | **`100%` (`14/14` products across `13/13` suppliers)** | **+37.5% recall (no `k=5` truncation)** |
