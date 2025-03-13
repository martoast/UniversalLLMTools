#!/usr/bin/env python3
"""
Simple example of using the provider-agnostic LLM tool system with a configuration file.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import our LLM tools library
from llm_tools import Provider, Tool, ToolCall, ToolResult, LLMToolClient


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and process the configuration file, replacing environment variables."""
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Replace environment variables
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        if placeholder in config_str:
            config_str = config_str.replace(placeholder, value)
    
    # Parse JSON
    return json.loads(config_str)


def get_llm_client(config: Dict[str, Any]) -> LLMToolClient:
    """Create an LLM client based on the configuration."""
    provider_name = config["llm_provider"].lower()
    provider_config = config["providers"][provider_name]
    
    # Map provider name to enum
    provider_map = {
        "anthropic": Provider.ANTHROPIC,
        "openai": Provider.OPENAI,
        "gemini": Provider.GOOGLE
    }
    
    if provider_name not in provider_map:
        raise ValueError(f"Unsupported provider: {provider_name}")
    
    provider = provider_map[provider_name]
    api_key = provider_config["api_key"]
    
    # Create the client
    return LLMToolClient(provider, api_key)


def define_simple_tools() -> List[Tool]:
    """Define a simple calculator tool as an example."""
    return [
        Tool(
            name="calculator",
            description="Perform a mathematical calculation with basic operations (+, -, *, /).",
            parameters=[
                {
                    "name": "expression",
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4').",
                    "required": True
                }
            ]
        ),
        Tool(
            name="notepad",
            description="Read from or write to a simple text notepad.",
            parameters=[
                {
                    "name": "action",
                    "type": "string",
                    "description": "Whether to 'read' from or 'write' to the notepad.",
                    "enum": ["read", "write"],
                    "required": True
                },
                {
                    "name": "content",
                    "type": "string",
                    "description": "The content to write (only needed for 'write' action).",
                    "required": False
                }
            ]
        )
    ]


def execute_calculator(expression: str) -> Dict[str, Any]:
    """Execute the calculator tool."""
    try:
        # WARNING: eval is used here for simplicity. In a real application,
        # you should use a safe expression evaluator.
        result = eval(expression, {"__builtins__": {}})
        return {
            "result": result,
            "expression": expression
        }
    except Exception as e:
        return {
            "error": str(e),
            "expression": expression
        }


def execute_notepad(action: str, content: Optional[str], notepad_path: str) -> Dict[str, Any]:
    """Execute the notepad tool."""
    path = Path(notepad_path)
    
    if action == "read":
        if path.exists():
            try:
                with open(path, 'r') as f:
                    content = f.read()
                return {
                    "content": content,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "error": f"Failed to read notepad: {str(e)}",
                    "status": "error"
                }
        else:
            return {
                "content": "",
                "status": "empty",
                "message": "Notepad is empty or does not exist."
            }
    
    elif action == "write":
        if not content:
            return {
                "error": "No content provided for write action.",
                "status": "error"
            }
        
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                f.write(content)
            
            return {
                "status": "success",
                "message": f"Successfully wrote {len(content)} characters to notepad."
            }
        except Exception as e:
            return {
                "error": f"Failed to write to notepad: {str(e)}",
                "status": "error"
            }
    
    else:
        return {
            "error": f"Invalid action: {action}. Must be 'read' or 'write'.",
            "status": "error"
        }


def execute_tool(tool_call: ToolCall, config: Dict[str, Any]) -> ToolResult:
    """Execute the tool call and return the result."""
    if tool_call.name == "calculator":
        expression = tool_call.arguments.get("expression", "")
        
        try:
            result = execute_calculator(expression)
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=result
            )
        except Exception as e:
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=f"Error executing calculator: {str(e)}",
                is_error=True
            )
    
    elif tool_call.name == "notepad":
        action = tool_call.arguments.get("action", "")
        content = tool_call.arguments.get("content", "")
        notepad_path = config["notepad_path"]
        
        try:
            result = execute_notepad(action, content, notepad_path)
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=result
            )
        except Exception as e:
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=f"Error executing notepad: {str(e)}",
                is_error=True
            )
    
    else:
        return ToolResult(
            id=tool_call.id,
            name=tool_call.name,
            output=f"Unknown tool: {tool_call.name}",
            is_error=True
        )


def run_conversation(config: Dict[str, Any], message: str):
    """Run a conversation with the LLM using tools."""
    # Create LLM client
    client = get_llm_client(config)
    
    # Define tools
    tools = define_simple_tools()
    
    # Get provider-specific options
    provider_name = config["llm_provider"].lower()
    provider_config = config["providers"][provider_name]
    
    options = {
        "model": provider_config["model_name"],
        "max_tokens": provider_config.get("max_tokens", 1024)
    }
    
    # Send message to LLM
    print(f"\n🔹 Sending message to {provider_name}:\n  \"{message}\"\n")
    
    response, tool_calls, text = client.call_with_tools(message, tools, **options)
    
    # Print any immediate text response
    if text:
        print(f"🤖 Initial Response:\n{text}\n")
    
    # Process tool calls if any
    if tool_calls:
        print(f"🛠️ Tool Calls ({len(tool_calls)}):")
        
        for i, tool_call in enumerate(tool_calls):
            print(f"\n  Tool Call #{i+1}:")
            print(f"  Name: {tool_call.name}")
            print(f"  Arguments: {json.dumps(tool_call.arguments, indent=2)}")
        
        # Execute tools and collect results
        tool_results = [execute_tool(tool_call, config) for tool_call in tool_calls]
        
        print("\n🔄 Tool Results:")
        for i, result in enumerate(tool_results):
            print(f"\n  Result #{i+1}:")
            print(f"  Tool: {result.name}")
            print(f"  Output: {json.dumps(result.output, indent=2) if isinstance(result.output, dict) else result.output}")
        
        # Submit tool results back to LLM
        print("\n🔹 Submitting tool results back to LLM...\n")
        
        # Add the original message to options for better context
        options["original_message"] = message
        
        response, text = client.submit_tool_results(tool_results, **options)
        
        print(f"🤖 Final Response:\n{text}")
    else:
        print("No tool calls were made by the LLM.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Tool Example with Config")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--message", 
        type=str,
        default="What is 25 * 16? Also, can you write 'Hello, world!' to the notepad and then read it back?",
        help="Message to send to the LLM"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the conversation
    run_conversation(config, args.message)


if __name__ == "__main__":
    main()