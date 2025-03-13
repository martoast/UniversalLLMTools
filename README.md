# UniversalLLMTools

A provider-agnostic tool use implementation for LLM APIs (Anthropic, Google, OpenAI).

## Overview

UniversalLLMTools provides a unified interface for creating and using tools with different LLM providers, allowing you to write code once and use it with any supported LLM. It abstracts away the differences between how Anthropic's Claude, Google's Gemini, and OpenAI's GPT models handle function/tool calling.

## Features

- Define tools once, use them with any provider
- Unified API for tool invocation and result handling
- Direct API access for Anthropic (avoiding SDK dependency issues)
- Support for multiple tools and multi-step tool interactions
- Consistent experience across providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UniversalLLMTools.git
cd UniversalLLMTools

# Install dependencies
pip install requests openai google-generativeai anthropic
```

## Usage

### Basic Example

```python
from llm_tools import Provider, Tool, LLMToolClient

# Define your tools once
calculator_tool = Tool(
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
)

notepad_tool = Tool(
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

# Create a client for your chosen provider
client = LLMToolClient(
    provider=Provider.OPENAI,  # or ANTHROPIC or GOOGLE
    api_key="your-api-key"
)

# Call the LLM with your tools
response, tool_calls, text = client.call_with_tools(
    message="What is 25 * 16? Also, can you write 'Hello, world!' to the notepad and then read it back?",
    tools=[calculator_tool, notepad_tool]
)

# Process any tool calls
if tool_calls:
    # Execute the tools (implementation depends on your use case)
    tool_results = []
    for tool_call in tool_calls:
        if tool_call.name == "calculator":
            # Execute the calculator
            result = execute_calculator(tool_call.arguments["expression"])
            tool_results.append(ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=result
            ))
        elif tool_call.name == "notepad":
            # Execute the notepad
            result = execute_notepad(
                action=tool_call.arguments["action"],
                content=tool_call.arguments.get("content")
            )
            tool_results.append(ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=result
            ))
    
    # Submit the tool results back to the LLM
    response, text = client.submit_tool_results(tool_results)
    print(f"LLM Response: {text}")
else:
    print(f"LLM Response: {text}")
```

### Configuration-Based Usage

You can also use a configuration file to specify the provider and API keys:

```python
import json
from llm_tools import Provider, Tool, LLMToolClient

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

provider_name = config["llm_provider"].lower()
provider_config = config["providers"][provider_name]

# Map provider name to enum
provider_map = {
    "anthropic": Provider.ANTHROPIC,
    "openai": Provider.OPENAI,
    "gemini": Provider.GOOGLE
}

provider = provider_map[provider_name]
api_key = provider_config["api_key"]

# Create client and use tools
client = LLMToolClient(provider, api_key)
```

## Provider-Specific Notes

### Anthropic (Claude)

- Uses a direct API implementation to avoid SDK dependency issues
- Requires a properly formatted system prompt for best results
- May require more explicit instructions to use multiple tools

### Google (Gemini)

- Handles MapComposite objects for function arguments
- Requires special handling for multiple tool results
- Uses the google.generativeai package

### OpenAI (GPT)

- Uses the standard Chat Completions API
- Requires proper message threading for tool results
- Uses the OpenAI Python SDK

## How It Works

1. **Define Tools**: Create Tool objects with consistent parameters
2. **Format Conversion**: The library converts your unified tool definition to provider-specific formats
3. **API Calls**: Makes the appropriate API calls for each provider
4. **Response Parsing**: Normalizes different response formats to a consistent structure
5. **Result Submission**: Handles the different ways each provider processes tool results

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.