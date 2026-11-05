# The Entry Point
"""
Main Entry Point for Supply Chain Risk Analysis Agent

This module serves as the application entry point for the Global Supply Chain Risk Agent.
It orchestrates the execution of the LangGraph agent to analyze supply chain risks by
querying the knowledge graph and vector stores.

The module performs the following functions:
- Initializes the agent from agent.py
- Defines and executes user queries about supply chain risk events
- Invokes the LangGraph agent with appropriate configuration
- Displays the full conversation flow including tool calls and responses
- Handles errors gracefully with informative messages for API failures and recursion limits

Key Features:
- Comprehensive error handling for API failures (OpenRouter service issues)
- Recursion limit management to prevent infinite agent loops
- Detailed conversation flow output showing all messages, tool calls, and responses
- Automatic content deduplication to handle LLM response repetition

Example Usage:
    The module is typically executed directly:
    $ python main.py
    
    This will run the agent with a predefined query about assessing the impact of
    supply chain disruptions at specific locations.

Error Handling:
    - ValueError: Handles API errors (e.g., 502 from OpenRouter service)
    - GraphRecursionError: Detects when agent is stuck in infinite loops
    - BadRequestError: Identifies model incompatibility issues (e.g., unsupported function calling)
    - Generic Exception: Catches and displays unexpected errors

Configuration:
    - Recursion limit is set to 50 to allow complex multi-step agent reasoning
    - Tokenizer parallelism is disabled to prevent warnings during forking
"""

import ast
import os
import re
import sys
import time
from collections import Counter

# Fix tokenizer warning about parallelism after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agent import agent, launch_risk_map_ui
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


DEFAULT_QUERY = (
    "Assess the impact of a strike (geopolitical risk, strike, riots, etc.) "
    "at all San Diego location on our supply chain. "
    "FYI: my print out viewing window is very small, so please make it easy to read."
)
DIVIDER = "=" * 60
SUB_DIVIDER = "-" * 60


def deduplicate_content(content: str) -> str:
    """
    Deduplicate content if it's repeated (some LLMs occasionally duplicate their responses).
    This happens when the LLM generates the same response twice in a row.
    """
    if isinstance(content, str) and len(content) > 200:
        # Get the first ~100 chars as a signature to search for duplicates
        signature = content[:100].strip()
        # Find where this signature appears again (indicating duplicate)
        second_occurrence = content.find(signature, 100)

        if second_occurrence > 0:
            # Check if the text from second occurrence onwards matches the beginning
            # This would indicate a full duplicate
            remaining = content[second_occurrence:]
            beginning = content[:second_occurrence]

            # If remaining is similar length and starts the same, it's likely a duplicate
            if len(remaining) > len(beginning) * 0.8 and remaining[:len(beginning)] == beginning:
                content = beginning.strip()
            elif remaining[:200] == beginning[:200]:  # First 200 chars match = likely duplicate
                content = beginning.strip()
    return content


def extract_thinking_and_clean_content(message: BaseMessage) -> tuple[str | None, str]:
    """
    Extract any chain-of-thought / reasoning trace from an AIMessage (either in
    additional_kwargs['reasoning'] / ['reasoning_content'] or inline <think> tags)
    and return (thinking_text, cleaned_content).
    """
    raw_content = getattr(message, "content", "") or ""
    thinking = None

    additional = getattr(message, "additional_kwargs", {}) or {}
    thinking = additional.get("reasoning_content") or additional.get("reasoning")

    if isinstance(raw_content, str) and "<think>" in raw_content and "</think>" in raw_content:
        match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted and not thinking:
                thinking = extracted
            raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    cleaned_content = deduplicate_content(raw_content).strip() if isinstance(raw_content, str) else str(raw_content).strip()
    return thinking, cleaned_content


def format_tool_output(content: str) -> str:
    """
    Format stringified Python lists from tool outputs into compact multi-line bullet points
    for easier reading in narrow terminal windows.
    """
    if not isinstance(content, str):
        return str(content)

    stripped = content.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                if not parsed:
                    return "  (empty list)"
                return "\n".join(f"  • {item}" for item in parsed)
        except (ValueError, SyntaxError):
            pass
    return f"  {stripped}"


def get_message_metadata_summary(message: BaseMessage) -> str | None:
    """
    Extract model name, token usage, and finish reason from AIMessage metadata if available.
    """
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


def run_agent_with_live_visibility(query: str, recursion_limit: int = 50) -> tuple[dict, float, list[dict]]:
    """
    Stream the LangGraph agent execution step-by-step so live progress is visible
    in real time, while collecting the final state and per-step timing telemetry.
    """
    # Configure recursion limit to prevent infinite loops
    # Increase if needed, but also indicates agent might be stuck
    config = {"recursion_limit": recursion_limit}
    inputs = {"messages": [HumanMessage(content=query)]}

    print("\n" + DIVIDER)
    print("LIVE AGENT EXECUTION STREAM")
    print(DIVIDER)

    start_time = time.perf_counter()
    step_start = start_time
    seen_count = 0
    step_index = 0
    step_metrics: list[dict] = []
    final_state: dict = {"messages": []}

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
            msg_type = msg.__class__.__name__

            if isinstance(msg, HumanMessage):
                print(f"[{total_elapsed:5.2f}s] Step {step_index}: 👤 HumanMessage queued")
            elif isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", []) or []
                if tool_calls:
                    calls_str = ", ".join(
                        f"{tc.get('name')}({tc.get('args', {})})" for tc in tool_calls
                    )
                    print(
                        f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                        f"🤖 Chatbot -> Calling {len(tool_calls)} tool(s): {calls_str}"
                    )
                else:
                    print(
                        f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                        f"🤖 Chatbot -> Generated final response ({len(msg.content or '')} chars)"
                    )
                step_metrics.append(
                    {"step": step_index, "type": msg_type, "elapsed": step_elapsed}
                )
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)
                if len(preview) > 75:
                    preview = preview[:72] + "..."
                print(
                    f"[{total_elapsed:5.2f}s] (+{step_elapsed:4.2f}s) Step {step_index}: "
                    f"🔧 ToolNode -> Result: {preview}"
                )
                step_metrics.append(
                    {"step": step_index, "type": msg_type, "elapsed": step_elapsed}
                )

        step_start = time.perf_counter()

    total_duration = time.perf_counter() - start_time
    return final_state, total_duration, step_metrics


def handle_execution_error(e: Exception) -> None:
    """
    Handle known OpenRouter, LangGraph recursion, and model compatibility errors gracefully.
    """
    if isinstance(e, ValueError):
        # Handle API errors (e.g., 502 from OpenRouter)
        error_msg = str(e)
        if "502" in error_msg or "Upstream error" in error_msg:
            print("\n⚠️  API Error: The OpenRouter service is temporarily unavailable.")
            print("   This is usually a temporary issue. Please try again in a few moments.")
            print(f"   Error details: {error_msg}")
            print("\n💡 Tip: Free models on OpenRouter may have rate limits or temporary outages.")
            print("   Consider trying a different model or waiting a few minutes.")
        else:
            print(f"\n❌ Error: {error_msg}")
        sys.exit(1)

    error_type = type(e).__name__
    error_msg = str(e)

    # Handle GraphRecursionError - agent stuck in loop
    if "GraphRecursionError" in error_type or "recursion limit" in error_msg.lower():
        print("\n⚠️  Recursion Limit Reached: The agent got stuck in a loop.")
        print("   This usually happens when:")
        print("   1. The location/entity name doesn't match exactly")
        print("   2. The LLM keeps making tool calls without finding answers")
        print("   3. Tool results aren't clear enough for the LLM to complete")
        print(f"\n   Error: {error_msg}")
        print("\n💡 Suggestions:")
        print("   1. Try a more specific location name (check available locations)")
        print("   2. Simplify your query")
        print("   3. The recursion limit can be increased, but may indicate a deeper issue")

    # Handle BadRequestError (400) - usually means model incompatibility
    elif "BadRequestError" in error_type or "400" in error_msg:
        print("\n⚠️  Model Compatibility Error: The selected model doesn't support the required API format.")
        print("   Some models on OpenRouter may not fully support function calling or tool use.")
        print(f"   Error: {error_msg}")
        print("\n💡 Suggestions:")
        print("   1. Try a different model that supports function calling")
        print("   2. Check OpenRouter docs: https://openrouter.ai/models?supported_parameters=tools")
        print("   3. Models known to work: openrouter/free, deepseek/deepseek-chat-v3:free, qwen/qwen-2.5-72b-instruct:free")
        print("\n   To change model, edit agent.py and update the model_name variable.")
    else:
        print(f"\n❌ Unexpected error: {error_type}: {error_msg}")
    sys.exit(1)


def print_conversation_flow(messages: list[BaseMessage], total_duration: float) -> None:
    """
    Display detailed conversation flow including messages, reasoning traces,
    linked tool calls/results, model metadata, and execution telemetry.
    """
    print("\n" + DIVIDER)
    print("AGENT CONVERSATION FLOW (DETAILED TRACE)")
    print(DIVIDER)

    # Build a lookup map of tool_call_id -> tool call details so ToolMessages show what called them
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

    # Print all messages to see the conversation flow
    print(f"\nTotal messages: {len(messages)}")
    print(f"Last message type: {messages[-1].__class__.__name__}\n")

    for i, message in enumerate(messages):
        msg_class = message.__class__.__name__
        role_badge = {
            "HumanMessage": "👤 HUMAN",
            "AIMessage": "🤖 AI (CHATBOT)",
            "ToolMessage": "🔧 TOOL OUTPUT",
        }.get(msg_class, msg_class)

        print(f"\n--- Message {i+1}/{len(messages)}: {role_badge} ({msg_class}) ---")

        # Surface LLM metadata (resolved model, finish reason, tokens) when available
        meta_summary = get_message_metadata_summary(message)
        if meta_summary:
            print(f"Meta: [{meta_summary}]")

        # If this is a ToolMessage, link it back to the originating tool call & arguments
        if hasattr(message, "tool_call_id") and message.tool_call_id:
            linked_call = tool_call_lookup.get(message.tool_call_id)
            if linked_call:
                print(
                    f"Source Call: {linked_call.get('name', 'unknown')}"
                    f"({linked_call.get('args', {})})"
                )
            print(f"Tool Call ID: {message.tool_call_id}")

        # Extract any chain-of-thought reasoning and deduplicate content
        thinking, content = extract_thinking_and_clean_content(message)
        if thinking:
            print(f"💭 Reasoning Trace:\n  {thinking}")

        # Handle different message types
        if hasattr(message, "content"):
            if content:
                if isinstance(message, ToolMessage):
                    print("Result:")
                    print(format_tool_output(content))
                else:
                    print(f"Content:\n{content}")
            else:
                # Message has no content - this might indicate the agent is waiting or stuck
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print("Content: <delegating to tool execution>")
                else:
                    print("Content: <empty or None>")
                    print(
                        "⚠️  Warning: AIMessage has no content and no tool calls - "
                        "this might indicate an issue"
                    )

        # Extract tool calls from AIMessage
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"Tool Calls Dispatched: {len(message.tool_calls)}")
            for tool_call in message.tool_calls:
                print(f"  - Tool: {tool_call.get('name', 'unknown')}")
                print(f"    Args: {tool_call.get('args', {})}")

    # Execution Telemetry Summary
    ai_turns = sum(1 for m in messages if isinstance(m, AIMessage))
    tool_Executions = sum(1 for m in messages if isinstance(m, ToolMessage))

    print("\n" + DIVIDER)
    print("EXECUTION TELEMETRY & TOOL SUMMARY")
    print(DIVIDER)
    print(f"⏱️  Total Runtime      : {total_duration:.2f}s")
    print(f"🔄 LLM Reasoning Turns: {ai_turns}")
    print(f"🛠️  Tool Executions    : {tool_Executions}")
    if models_used:
        print(f"🧠 Model(s) Routed    : {', '.join(sorted(models_used))}")
    if total_tokens > 0:
        print(f"📊 Total Tokens Used  : {total_tokens:,}")
    if tool_counter:
        print("📋 Tool Breakdown     :")
        for tool_name, count in tool_counter.most_common():
            print(f"   • {tool_name}: {count}x")

    # Check if we have a final response
    print("\n" + DIVIDER)
    print("FINAL STATUS")
    print(DIVIDER)
    last_message = messages[-1]
    if hasattr(last_message, "content") and last_message.content:
        print("✅ Agent completed successfully with final response")
    else:
        print("⚠️  Agent completed but final message has no content")
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"   Last message has {len(last_message.tool_calls)} pending tool call(s)")
        print("   This may indicate:")
        print("   - Agent hit recursion limit (check for GraphRecursionError)")
        print("   - LLM didn't generate a final response")
        print("   - Agent is waiting for tool execution")


def main(query: str = DEFAULT_QUERY) -> None:
    print("Query:")
    print(query)
    print(DIVIDER)
    print(launch_risk_map_ui())

    try:
        result, total_duration, _ = run_agent_with_live_visibility(query, recursion_limit=50)
    except Exception as e:
        handle_execution_error(e)
        return

    print_conversation_flow(result["messages"], total_duration)


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_QUERY
    main(user_query)


