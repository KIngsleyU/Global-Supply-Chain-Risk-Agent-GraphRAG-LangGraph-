"""
Evaluation & Stress-Testing Suite for the Supply Chain Risk Agent.

Runs the full LangGraph agent 3 times across distinct supply-chain risk scenarios
and compares its outputs against the exact in-memory NetworkX ground truth.

Can evaluate either:
- Baseline agent (`agent.py` in repo root) — default
- Phase 1 upgraded agent (`agentic_robustness_review/agent_phase1.py`) via `--phase1` flag
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path when running from inside agentic_robustness_review/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import networkx as nx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from schema import Location, Product, Supplier


def compute_ground_truth(graph_db):
    """Extract ground-truth stats and full hierarchy from the in-memory graph_db."""
    g = graph_db.graph
    locations = graph_db.locations
    suppliers = graph_db.suppliers
    products = graph_db.products

    by_country = {}
    for loc in locations:
        by_country.setdefault(loc.country, []).append(loc)

    loc_hierarchy = {}
    for loc in locations:
        loc_sups = [
            n for n in nx.all_neighbors(g, loc) if isinstance(n, Supplier)
        ]
        sup_map = {}
        for s in loc_sups:
            s_prods = [
                p for p in nx.all_neighbors(g, s) if isinstance(p, Product)
            ]
            sup_map[s.name] = {
                "risk_score": s.risk_score,
                "revenue": s.revenue,
                "products": [
                    {"name": p.name, "sku": p.sku, "price": p.price}
                    for p in s_prods
                ],
            }
        loc_hierarchy[loc.name] = {
            "country": loc.country,
            "suppliers": sup_map,
        }

    return {
        "num_locations": len(locations),
        "num_suppliers": len(suppliers),
        "num_products": len(products),
        "countries": {k: [l.name for l in v] for k, v in by_country.items()},
        "hierarchy": loc_hierarchy,
    }


def run_single_experiment(target_agent, run_id: int, name: str, query: str):
    print(f"\n{'=' * 70}")
    print(f"RUN {run_id}: {name}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}")

    start = time.perf_counter()
    config = {"recursion_limit": 50}
    inputs = {"messages": [HumanMessage(content=query)]}

    step_num = 0
    seen = 0
    final_state = None
    step_timings = []
    step_start = start

    try:
        for state in target_agent.stream(inputs, config=config, stream_mode="values"):
            now = time.perf_counter()
            step_elapsed = now - step_start
            total_elapsed = now - start
            msgs = state.get("messages", [])
            new_msgs = msgs[seen:]
            seen = len(msgs)
            final_state = state

            for m in new_msgs:
                step_num += 1
                if isinstance(m, AIMessage):
                    resp_meta = getattr(m, "response_metadata", {}) or {}
                    model_used = resp_meta.get("model_name") or resp_meta.get("model") or "unknown"
                    tcs = getattr(m, "tool_calls", []) or []
                    if tcs:
                        tc_desc = ", ".join(f"{tc.get('name')}({tc.get('args')})" for tc in tcs)
                        print(f"  [{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) AI ({model_used}) -> ToolCalls: {tc_desc}")
                    else:
                        print(f"  [{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) AI ({model_used}) -> Final Answer ({len(m.content or '')} chars)")
                    step_timings.append({
                        "step": step_num,
                        "role": "AI",
                        "model": model_used,
                        "elapsed": round(step_elapsed, 2),
                        "tool_calls": len(tcs),
                    })
                elif isinstance(m, ToolMessage):
                    preview = str(m.content)[:90].replace("\n", " ")
                    print(f"  [{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) ToolResult -> {preview}...")
                    step_timings.append({
                        "step": step_num,
                        "role": "Tool",
                        "elapsed": round(step_elapsed, 2),
                    })
            step_start = time.perf_counter()

        duration = round(time.perf_counter() - start, 2)
        messages = final_state["messages"]
        final_answer = messages[-1].content if messages else ""

        tool_calls_made = []
        models_routed = []
        for m in messages:
            if isinstance(m, AIMessage):
                rm = getattr(m, "response_metadata", {}) or {}
                if rm.get("model_name"):
                    models_routed.append(rm["model_name"])
                for tc in getattr(m, "tool_calls", []) or []:
                    tool_calls_made.append({"name": tc.get("name"), "args": tc.get("args")})

        print(f"\n--- Run {run_id} Final Answer ---")
        print(final_answer)
        return {
            "run_id": run_id,
            "name": name,
            "query": query,
            "status": "success",
            "duration_sec": duration,
            "total_messages": len(messages),
            "tool_calls": tool_calls_made,
            "models_routed": models_routed,
            "step_timings": step_timings,
            "final_answer": final_answer,
        }
    except Exception as e:
        duration = round(time.perf_counter() - start, 2)
        print(f"\n❌ Run {run_id} Failed after {duration}s: {type(e).__name__}: {e}")
        return {
            "run_id": run_id,
            "name": name,
            "query": query,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "duration_sec": duration,
        }


def main():
    parser = argparse.ArgumentParser(description="Run 3-trial supply chain agent evaluation.")
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Evaluate the Phase 1 upgraded agent instead of the baseline agent.",
    )
    args = parser.parse_args()

    if args.phase1:
        print("🚀 Loading Phase 1 Upgraded Agent (agentic_robustness_review/agent_phase1.py)...")
        from agentic_robustness_review.agent_phase1 import agent as target_agent, graph_db
        output_filename = Path(__file__).resolve().parent / "eval_results_phase1.json"
    else:
        print("📦 Loading Baseline Agent (agent.py)...")
        from agent import agent as target_agent, graph_db
        output_filename = Path(__file__).resolve().parent / "eval_results_baseline.json"

    gt = compute_ground_truth(graph_db)
    sample_loc = graph_db.locations[0]
    city_keyword = (
        sample_loc.name.split()[0]
        if sample_loc.name.split()[0] != "Port"
        else sample_loc.name.split()[-1]
    )

    queries = [
        (
            "Run 1: Standard Regional Disruption (Baseline Multi-Hop Traversal)",
            f"Assess the impact of a strike (geopolitical risk, strike, riots, etc.) at all {city_keyword} locations on our supply chain. "
            f"List every affected location, supplier (with risk score & revenue), and downstream product (with SKU & price). "
            f"FYI: my print out viewing window is very small, so please make it easy to read.",
        ),
        (
            "Run 2: Misspelled / Fuzzy Entity + Multi-Location Country Aggregation",
            "We have a major customs freeze across all ports and facilities in Chna (especially shanghia and shenzen). "
            "Identify all affected locations, their suppliers, the exact products at risk, and calculate the total supplier revenue exposed.",
        ),
        (
            "Run 3: Reverse Product-to-Location High-Risk Audit (Upstream Traversal)",
            "Find our Pharmaceutical or medical products (like syringes, gloves, masks, or vaccine carriers), "
            "check which suppliers manufacture them, identify where those suppliers are located, and flag any supplier with risk_score > 0.6.",
        ),
    ]

    results = []
    for idx, (title, q) in enumerate(queries, start=1):
        res = run_single_experiment(target_agent, idx, title, q)
        results.append(res)

    output_payload = {
        "agent_version": "phase1" if args.phase1 else "baseline",
        "ground_truth": gt,
        "runs": results,
    }
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n✅ Saved detailed evaluation telemetry to {output_filename}")


if __name__ == "__main__":
    main()

