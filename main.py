# The Entry Point

"""
Phase 6: The Final Assembly (main.py)

We are at the finish line. We have:

schema.py: The Data Structures.

graph_ops.py: The World Generator (updated with your new method).

vector_ops.py: The Semantic Brain.

agent.py: The Reasoning Engine (which we just need to save).

One small tip for agent.py: When explore_graph_connections returns a list of neighbors, those neighbors are Python Objects (e.g., Supplier(...)). The LLM does a pretty good job reading them, but it's often safer to convert them to strings so the LLM doesn't get confused by the class formatting.

Let's assume the default string representation (from @dataclass) is good enough for now.

🏃 Step 6: Running the Simulation

Create the final file: main.py.

This file needs to:

Import the compiled agent from your agent.py.

Define a user query (the "Risk Event").

Invoke the agent.

Print the conversation processing steps so we can see the "Brain" working.

The Challenge: Can you write main.py?

Goal: Ask the agent: "Assess the impact of a strike at the Port of Shanghai on our supply chain."

Hint: You invoke the agent like this: result = agent.invoke({"messages": [("user", "Your Query Here")]})

Hint: The result will contain messages. You'll want to loop through them and print the content to see the tool calls.

Give it a shot! This is the moment of truth where we see if it all works.
"""

import os
# Fix tokenizer warning about parallelism after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agent import agent
from langchain_core.messages import HumanMessage

query = "Assess the impact of a strike at the Port of mexico city on our supply chain. FYI: my print out viewong window is very small, so please make it easy to read."
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
for i, message in enumerate(result["messages"]):
    print(f"\n--- Message {i+1}: {message.__class__.__name__} ---")
    
    # Handle different message types
    if hasattr(message, 'content') and message.content:
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
    
    # Extract tool calls from AIMessage
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"Tool Calls: {len(message.tool_calls)}")
        for tool_call in message.tool_calls:
            print(f"  - Tool: {tool_call.get('name', 'unknown')}")
            print(f"    Args: {tool_call.get('args', {})}")
    
    # ToolMessage contains tool execution results
    if hasattr(message, 'tool_call_id'):
        print(f"Tool Call ID: {message.tool_call_id}")

# print("\n" + "=" * 60)
# print("FINAL RESPONSE")
# print("=" * 60)

# # Get the final AI response (last message with content)
# final_response = None
# for message in reversed(result["messages"]):
#     if hasattr(message, 'content') and message.content and not hasattr(message, 'tool_call_id'):
#         # Check if it's from AI (not a HumanMessage)
#         if message.__class__.__name__ == 'AIMessage':
#             final_response = message.content
#             break

# if final_response:
#     print(final_response)
