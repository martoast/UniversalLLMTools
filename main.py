#!/usr/bin/env python3
"""
Example usage of the LLM tool system
"""
import json
import argparse
from llm_tools import LLMToolClient, Tool, Provider, execute_calculator

def main():
    parser = argparse.ArgumentParser(description="LLM Tool Example")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--message", 
        type=str,
        default="What is 512 * 48?",
        help="Message to send"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Create client
    provider = Provider(config["llm_provider"])
    api_key = config["providers"][provider]["api_key"]
    client = LLMToolClient(provider, api_key)

    # Define calculator tool
    calculator = Tool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters=[{
            "name": "expression",
            "type": "string",
            "description": "Math expression to evaluate",
            "required": True
        }]
    )

    # Send request
    print(f"\n🔹 Sending message to {provider}:\n  \"{args.message}\"\n")
    
    response, tool_calls, text = client.call_with_tools(
        message=args.message,
        tools=[calculator],
        model=config["providers"][provider]["model_name"]
    )
    
    print(f"🤖 Initial Response:\n{text}\n")
    
    if tool_calls:
        print(f"🛠️ Tool Calls ({len(tool_calls)}):")
        for i, call in enumerate(tool_calls):
            print(f"\n  Call #{i+1}:")
            print(f"  Name: {call.name}")
            print(f"  Arguments: {json.dumps(call.arguments, indent=2)}")
            
            if call.name == "calculator":
                result = execute_calculator(call.arguments.get("expression", ""))
                print(f"\n  Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()