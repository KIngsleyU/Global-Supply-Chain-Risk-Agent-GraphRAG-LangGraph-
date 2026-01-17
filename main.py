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

import os
# Fix tokenizer warning about parallelism after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agent import agent
from langchain_core.messages import HumanMessage

query = "Assess the impact of a strike at the Port of mexico city on our supply chain. FYI: my print out viewing window is very small, so please make it easy to read."
print("Query:")
print(query)
print("=" * 60)

try:
    # Configure recursion limit to prevent infinite loops
    # Increase if needed, but also indicates agent might be stuck
    config = {"recursion_limit": 50}
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
except ValueError as e:
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
    exit(1)
except Exception as e:
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
        print("   3. Models known to work: deepseek/deepseek-chat-v3:free, qwen/qwen-2.5-72b-instruct:free")
        print("\n   To change model, edit agent.py and update the model_name variable.")
    else:
        print(f"\n❌ Unexpected error: {error_type}: {error_msg}")
    exit(1)

print("\n" + "=" * 60)
print("AGENT CONVERSATION FLOW")
print("=" * 60)

# Print all messages to see the conversation flow
print(f"\nTotal messages: {len(result['messages'])}")
print(f"Last message type: {result['messages'][-1].__class__.__name__}\n")

for i, message in enumerate(result["messages"]):
    print(f"\n--- Message {i+1}: {message.__class__.__name__} ---")
    
    # Handle different message types
    if hasattr(message, 'content'):
        if message.content:
            content = message.content
            # Deduplicate content if it's repeated (some LLMs occasionally duplicate their responses)
            # This happens when the LLM generates the same response twice in a row
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
            
            print(f"Content: {content}")
        else:
            # Message has no content - this might indicate the agent is waiting or stuck
            print(f"Content: <empty or None>")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # print(f"⚠️  Warning: AIMessage has no content but has tool calls - agent may be in progress")
                pass
            else:
                print(f"⚠️  Warning: AIMessage has no content and no tool calls - this might indicate an issue")
    
    # Extract tool calls from AIMessage
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"Tool Calls: {len(message.tool_calls)}")
        for tool_call in message.tool_calls:
            print(f"  - Tool: {tool_call.get('name', 'unknown')}")
            print(f"    Args: {tool_call.get('args', {})}")
    
    # ToolMessage contains tool execution results
    if hasattr(message, 'tool_call_id'):
        print(f"Tool Call ID: {message.tool_call_id}")

# Check if we have a final response
print("\n" + "=" * 60)
print("FINAL STATUS")
print("=" * 60)
last_message = result["messages"][-1]
if hasattr(last_message, 'content') and last_message.content:
    print("✅ Agent completed successfully with final response")
else:
    print("⚠️  Agent completed but final message has no content")
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"   Last message has {len(last_message.tool_calls)} pending tool call(s)")
    print("   This may indicate:")
    print("   - Agent hit recursion limit (check for GraphRecursionError)")
    print("   - LLM didn't generate a final response")
    print("   - Agent is waiting for tool execution")

