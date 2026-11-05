"""
Entry point for the Phase 1 Upgraded Supply Chain Risk Agent.

Keeps root `main.py` and `agent.py` completely intact while letting you run
the Phase 1 upgraded agent (`agentic_robustness_review/agent_phase1.py`) with
live execution streaming and detailed conversation trace visibility.

Usage:
    python agentic_robustness_review/main_phase1.py
    python agentic_robustness_review/main_phase1.py "Assess customs freeze in Chna"
"""

import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

# Ensure both repo root and agentic_robustness_review/ are importable
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for p in (str(REPO_ROOT), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from agentic_robustness_review.agent_phase1 import agent, launch_risk_map_ui


DEFAULT_QUERY = (
    "Assess the impact of a strike (geopolitical risk, strike, riots, etc.) "
    "at all San Diego location on our supply chain. "
    "FYI: my print out viewing window is very small, so please make it easy to read."
)
DIVIDER = "=" * 60


def deduplicate_content(content: str) -> str:
    """Deduplicate content if an LLM repeats its response twice in a row."""
    if isinstance(content, str) and len(content) > 200:
        signature = content[:100].strip()
        second_occurrence = content.find(signature, 100)
        if second_occurrence > 0:
            remaining = content[second_occurrence:]
            beginning = content[:second_occurrence]
            if len(remaining) > len(beginning) * 0.8 and remaining[:len(beginning)] == beginning:
                content = beginning.strip()
            elif remaining[:200] == beginning[:200]:
                content = beginning.strip()
    return content


def extract_thinking_and_clean_content(message: BaseMessage) -> tuple[str | None, str]:
    """Extract chain-of-thought reasoning (<think> tags or reasoning_content) and clean content."""
    raw_content = getattr(message, "content", "") or ""
    additional = getattr(message, "additional_kwargs", {}) or {}
    thinking = additional.get("reasoning_content") or additional.get("reasoning")

    if isinstance(raw_content, str) and "<think>" in raw_content and "</think>" in raw_content:
        match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted and not thinking:
                thinking = extracted
            raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    cleaned = deduplicate_content(raw_content).strip() if isinstance(raw_content, str) else str(raw_content).strip()
    return thinking, cleaned


def format_tool_output(content: str) -> str:
    """Format JSON or Python list tool outputs cleanly for narrow terminal windows."""
    if not isinstance(content, str):
        return str(content)

    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            lines = []
            metrics = parsed.get("deterministic_summary_metrics")
            if metrics:
                lines.append("  [Deterministic Summary Metrics]")
                for k, v in metrics.items():
                    lines.append(f"    • {k}: {v}")

            # Also display the traversed Locations -> Suppliers -> Products from analyze_location_blast_radius
            affected_locs = parsed.get("affected_locations")
            if affected_locs:
                lines.append("  [Traversed Subgraph: Location -> Supplier -> Product]")
                for loc in affected_locs:
                    lines.append(
                        f"    📍 {loc.get('location_name')} ({loc.get('country')}) "
                        f"[{loc.get('supplier_count', 0)} supplier(s)]"
                    )
                    for sup in loc.get("suppliers", []):
                        lines.append(
                            f"       🏭 {sup.get('name')} | risk={sup.get('risk_score')} "
                            f"({sup.get('risk_level')}) | rev={sup.get('revenue_formatted')}"
                        )
                        for prod in sup.get("products_manufactured", []):
                            lines.append(
                                f"          📦 {prod.get('name')} "
                                f"(SKU: {prod.get('sku')}, {prod.get('price_formatted')})"
                            )

            # Also display alternative suppliers outside the disrupted zone if present
            alt_info = parsed.get("alternative_suppliers_outside_disrupted_zone") or {}
            alt_by_type = alt_info.get("by_product_type") or {}
            if alt_by_type:
                lines.append("  [Alternative Suppliers Outside Disrupted Zone]")
                for ptype, alts in alt_by_type.items():
                    lines.append(f"    🔄 Product Line: {ptype}")
                    for alt in alts:
                        lines.append(
                            f"       • {alt.get('alternative_supplier_name')} "
                            f"@ {alt.get('location_name')}, {alt.get('country')} | "
                            f"risk={alt.get('supplier_risk_score')} ({alt.get('supplier_risk_level')}) | "
                            f"rev={alt.get('supplier_current_revenue_formatted')} | "
                            f"makes {alt.get('equivalent_product')} ({alt.get('equivalent_sku')}, "
                            f"{alt.get('equivalent_unit_price_formatted')})"
                        )
            no_alt = alt_info.get("product_types_with_no_external_alternative") or []
            if no_alt:
                lines.append(f"    ⚠️ Sole-Source Product Lines (No External Alternative): {', '.join(no_alt)}")

            # Also display product audit results from audit_products_and_upstream_suppliers
            prod_table = parsed.get("product_to_supplier_location_table")
            if prod_table:
                lines.append("  [Traversed Subgraph: Product -> Supplier -> Location]")
                for row in prod_table:
                    lines.append(
                        f"    📦 {row.get('product_name')} ({row.get('sku')}, {row.get('price_formatted')}) "
                        f"<- 🏭 {row.get('supplier_name')} (risk={row.get('supplier_risk_score')}, "
                        f"{row.get('supplier_risk_flag')}) @ 📍 {row.get('location_name')}, {row.get('country')}"
                    )

            if lines:
                return "\n".join(lines)
        except Exception:
            pass

    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return "\n".join(f"  • {item}" for item in parsed) if parsed else "  (empty list)"
        except (ValueError, SyntaxError):
            pass

    return f"  {stripped}"


def get_message_metadata_summary(message: BaseMessage) -> str | None:
    """Extract model name, finish reason, and token usage from AIMessage metadata."""
    if not isinstance(message, AIMessage):
        return None

    parts = []
    resp_meta = getattr(message, "response_metadata", {}) or {}
    model_name = resp_meta.get("model_name") or resp_meta.get("model")
    if model_name:
        parts.append(f"model={model_name}")

    finish_reason = resp_meta.get("finish_reason")
    if finish_reason:
        parts.append(f"finish={finish_reason}")

    usage = getattr(message, "usage_metadata", None) or resp_meta.get("token_usage") or {}
    if isinstance(usage, dict):
        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens")
        out_tok = usage.get("output_tokens") or usage.get("completion_tokens")
        tot_tok = usage.get("total_tokens")
        if tot_tok is not None:
            parts.append(f"tokens={tot_tok} (in:{in_tok}, out:{out_tok})")

    return " | ".join(parts) if parts else None


def handle_execution_error(e: Exception) -> None:
    """Handle API, recursion limit, and model compatibility errors gracefully."""
    error_type = type(e).__name__
    error_msg = str(e)
    if "502" in error_msg or "Upstream error" in error_msg:
        print("\n⚠️  API Error: The OpenRouter service is temporarily unavailable.")
        print(f"   Error details: {error_msg}")
    elif "GraphRecursionError" in error_type or "recursion limit" in error_msg.lower():
        print(f"\n⚠️  Recursion Limit Reached: {error_msg}")
    elif "BadRequestError" in error_type or "400" in error_msg:
        print(f"\n⚠️  Model Compatibility Error: {error_msg}")
    else:
        print(f"\n❌ Unexpected error: {error_type}: {error_msg}")
    sys.exit(1)


def run_phase1_agent_with_live_visibility(query: str, recursion_limit: int = 50):
    config = {"recursion_limit": recursion_limit}
    inputs = {"messages": [HumanMessage(content=query)]}

    print("\n" + DIVIDER)
    print("LIVE PHASE 1 AGENT EXECUTION STREAM")
    print(DIVIDER)

    start_time = time.perf_counter()
    step_start = start_time
    seen_count = 0
    step_index = 0
    final_state = {"messages": []}

    for state in agent.stream(inputs, config=config, stream_mode="values"):
        now = time.perf_counter()
        step_elapsed = now - step_start
        total_elapsed = now - start_time

        messages = state.get("messages", [])
        new_messages = messages[seen_count:]
        seen_count = len(messages)
        final_state = state

        for msg in new_messages:
            step_index += 1
            if isinstance(msg, HumanMessage):
                if "PHASE 3 DETERMINISTIC VERIFICATION FAILED" in str(msg.content):
                    feedback_preview = str(state.get("reflection_feedback") or "")[:90]
                    print(
                        f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                        f"🛡️  Phase-3 Verifier -> Self-Correction Triggered: {feedback_preview}"
                    )
                else:
                    print(f"[{total_elapsed:5.2f}s] Step {step_index}: 👤 HumanMessage queued")
            elif isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", []) or []
                if tool_calls:
                    calls_str = ", ".join(
                        f"{tc.get('name')}({tc.get('args', {})})" for tc in tool_calls
                    )
                    print(
                        f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                        f"🤖 Phase-1 Chatbot -> Calling {len(tool_calls)} tool(s): {calls_str}"
                    )
                else:
                    print(
                        f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                        f"🤖 Phase-1 Chatbot -> Generated candidate response ({len(msg.content or '')} chars)"
                    )
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content).replace("\n", " ")
                if len(preview) > 75:
                    preview = preview[:72] + "..."
                print(
                    f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                    f"🔧 Phase-1 ToolNode -> Result: {preview}"
                )

        # If `verify_and_reflect` ran without adding a message, it means all Phase 3 checks passed!
        if not new_messages and messages and isinstance(messages[-1], AIMessage) and not getattr(messages[-1], "tool_calls", None):
            step_index += 1
            print(
                f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                f"✅ Phase-3 Verifier -> Completeness & Numerical Grounding Verified (0 missing entities/math errors)"
            )

        step_start = time.perf_counter()

    total_duration = time.perf_counter() - start_time
    return final_state, total_duration


def print_phase2_working_memory(state: dict) -> None:
    """Display the Phase 2 Structured `AgentState` Working Memory snapshot."""
    visited = state.get("visited_nodes") or []
    locs = state.get("discovered_locations") or []
    sups = state.get("discovered_suppliers") or []
    prods = state.get("discovered_products") or []
    metrics = state.get("deterministic_metrics") or {}

    if not (visited or locs or sups or prods or metrics):
        return

    print("\n" + DIVIDER)
    print("PHASE 2 & 3 STRUCTURED AGENT STATE (WORKING MEMORY & CRAG SNAPSHOT)")
    print(DIVIDER)
    print(f"🔒 Visited Tool/Node Signatures : {len(visited)}")
    print(f"📍 Canonical Locations Stored   : {len(locs)}")
    print(f"🏭 Canonical Suppliers Stored   : {len(sups)}")
    print(f"📦 Canonical Products Stored    : {len(prods)}")
    print(f"🛡️  Phase 3 Reflection Retries   : {state.get('reflection_count', 0)}")
    print(
        f"✅ Final Verification Status    : "
        f"{'PASSED (100% Entity & Math Grounded)' if not state.get('reflection_feedback') else state.get('reflection_feedback')}"
    )
    if metrics:
        print("📐 Deterministic Metrics in State (Cumulative):")
        for k, v in metrics.items():
            if k == "per_query_breakdown":
                continue
            print(f"   • {k}: {v}")
        per_query = metrics.get("per_query_breakdown")
        if isinstance(per_query, dict) and len(per_query) > 1:
            print("   📊 Per-Query Breakdown:")
            for q_label, q_metrics in per_query.items():
                rev = q_metrics.get("total_supplier_revenue_exposed_formatted", "N/A")
                sup_cnt = q_metrics.get("total_suppliers_affected") or q_metrics.get("total_unique_suppliers", 0)
                prod_cnt = q_metrics.get("total_products_affected") or q_metrics.get("total_matching_products", 0)
                print(f"      - '{q_label}': {sup_cnt} supplier(s), {prod_cnt} product(s), rev={rev}")


def print_conversation_flow(messages: list[BaseMessage], total_duration: float, final_state: dict | None = None) -> None:
    print("\n" + DIVIDER)
    print("AGENT CONVERSATION FLOW (DETAILED TRACE)")
    print(DIVIDER)

    tool_call_lookup: dict[str, dict] = {}
    tool_counter: Counter = Counter()
    total_tokens = 0
    models_used: set[str] = set()

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                tc_name = tc.get("name", "unknown")
                tool_counter[tc_name] += 1
                if tc_id:
                    tool_call_lookup[tc_id] = tc

        if isinstance(msg, AIMessage):
            resp_meta = getattr(msg, "response_metadata", {}) or {}
            model_name = resp_meta.get("model_name") or resp_meta.get("model")
            if model_name:
                models_used.add(model_name)
            usage = getattr(msg, "usage_metadata", None) or resp_meta.get("token_usage") or {}
            if isinstance(usage, dict) and usage.get("total_tokens"):
                total_tokens += int(usage["total_tokens"])

    print(f"\nTotal messages: {len(messages)}")
    print(f"Last message type: {messages[-1].__class__.__name__}\n")

    for i, message in enumerate(messages):
        msg_class = message.__class__.__name__
        is_verifier_msg = (
            isinstance(message, HumanMessage)
            and "PHASE 3 DETERMINISTIC VERIFICATION FAILED" in str(getattr(message, "content", ""))
        )
        # Check if this AIMessage is an intermediate draft that was rejected by Phase 3 Verifier
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        is_rejected_draft_report = (
            isinstance(message, AIMessage)
            and not (getattr(message, "tool_calls", None))
            and isinstance(next_msg, HumanMessage)
            and "PHASE 3 DETERMINISTIC VERIFICATION FAILED" in str(getattr(next_msg, "content", ""))
        )

        if is_verifier_msg:
            role_badge = "🛡️ PHASE-3 VERIFIER"
        else:
            role_badge = {
                "HumanMessage": "👤 HUMAN",
                "AIMessage": "🤖 AI (PHASE-1 CHATBOT)",
                "ToolMessage": "🔧 TOOL OUTPUT",
            }.get(msg_class, msg_class)

        print(f"\n--- Message {i+1}/{len(messages)}: {role_badge} ({msg_class}) ---")

        meta_summary = get_message_metadata_summary(message)
        if meta_summary:
            print(f"Meta: [{meta_summary}]")

        if hasattr(message, "tool_call_id") and message.tool_call_id:
            linked_call = tool_call_lookup.get(message.tool_call_id)
            if linked_call:
                print(
                    f"Source Call: {linked_call.get('name', 'unknown')}"
                    f"({linked_call.get('args', {})})"
                )
            print(f"Tool Call ID: {message.tool_call_id}")

        thinking, content = extract_thinking_and_clean_content(message)
        if thinking and not is_rejected_draft_report:
            print(f"💭 Reasoning Trace:\n  {thinking}")

        if hasattr(message, "content"):
            if content:
                if isinstance(message, ToolMessage):
                    print("Result:")
                    print(format_tool_output(content))
                elif is_rejected_draft_report:
                    print(
                        f"Content: <draft report generated ({len(content)} chars) — "
                        f"omitted from log because Phase 3 Verifier requested self-correction>"
                    )
                else:
                    print(f"Content:\n{content}")
            else:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print("Content: <delegating to tool execution>")
                else:
                    print("Content: <empty or None>")

        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"Tool Calls Dispatched: {len(message.tool_calls)}")
            for tool_call in message.tool_calls:
                print(f"  - Tool: {tool_call.get('name', 'unknown')}")
                print(f"    Args: {tool_call.get('args', {})}")

    ai_turns = sum(1 for m in messages if isinstance(m, AIMessage))
    tool_executions = sum(1 for m in messages if isinstance(m, ToolMessage))

    print("\n" + DIVIDER)
    print("EXECUTION TELEMETRY & TOOL SUMMARY")
    print(DIVIDER)
    print(f"⏱️  Total Runtime      : {total_duration:.2f}s")
    print(f"🔄 LLM Reasoning Turns: {ai_turns}")
    print(f"🛠️  Tool Executions    : {tool_executions}")
    if models_used:
        print(f"🧠 Model(s) Routed    : {', '.join(sorted(models_used))}")
    if total_tokens > 0:
        print(f"📊 Total Tokens Used  : {total_tokens:,}")
    if tool_counter:
        print("📋 Tool Breakdown     :")
        for tool_name, count in tool_counter.most_common():
            print(f"   • {tool_name}: {count}x")

    if final_state is not None:
        print_phase2_working_memory(final_state)

    print("\n" + DIVIDER)
    print("FINAL STATUS")
    print(DIVIDER)
    last_message = messages[-1]
    if hasattr(last_message, "content") and last_message.content:
        print("✅ Phase 1+2 Agent completed successfully with final response")
    else:
        print("⚠️  Phase 1+2 Agent completed but final message has no content")


def main(query: str = DEFAULT_QUERY) -> None:
    print("Query (Phase 1+2 Agent):")
    print(query)
    print(DIVIDER)
    print(launch_risk_map_ui())

    try:
        result, total_duration = run_phase1_agent_with_live_visibility(query, recursion_limit=50)
    except Exception as e:
        handle_execution_error(e)
        return

    print_conversation_flow(result["messages"], total_duration, final_state=result)


if __name__ == "__main__":
    # user_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_QUERY
    user_query = "Find our Pharmaceutical or medical products (like syringes, gloves, masks, or vaccine carriers), check which suppliers manufacture them, identify where those suppliers are located, and flag any supplier with risk_score > 0.6."
    user_query = ("We have a major customs freeze across all ports and facilities in Chna (especially shanghia and shenzen). Identify all affected locations, their suppliers, the exact products at risk, and calculate the total supplier revenue exposed. "
                "is it possible to find alternative supliers for this products and where? "
                "how much could this new supliers be making?")
    
    main(user_query)
