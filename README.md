# UniversalLLMTools

A simple tool system that works with Anthropic, OpenAI, and Google LLMs. Write your tool once, use it everywhere!

## Quick Start

1. Install dependencies:
```bash
pip install requests openai google-generativeai
pip install grpcio==1.60.1  # To avoid Google warning
```

2. Create your config.json:
```json
{
    "llm_provider": "anthropic",  // or "openai" or "google"
    "providers": {
        "anthropic": {
            "api_key": "your-anthropic-key",
            "model_name": "claude-3-sonnet-20240229"
        },
        "openai": {
            "api_key": "your-openai-key",
            "model_name": "gpt-4"
        },
        "google": {
            "api_key": "your-google-key",
            "model_name": "gemini-pro"
        }
    }
}
```

3. Run the example:
```bash
python main.py --message "What is 512 * 48?"
```

## How It Works

1. Define your tool once:
```python
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
```

2. Use it with any provider:
```python
# Load config and create client
with open("config.json") as f:
    config = json.load(f)

client = LLMToolClient(
    provider=Provider(config["llm_provider"]),
    api_key=config["providers"][config["llm_provider"]]["api_key"]
)

# Make request
response, tool_calls, text = client.call_with_tools(
    message="What is 512 * 48?",
    tools=[calculator]
)

# Handle tool calls
if tool_calls:
    for call in tool_calls:
        if call.name == "calculator":
            result = execute_calculator(call.arguments["expression"])
            print(f"Result: {result}")
```

## Supported Providers

- **Anthropic (Claude)**: Direct API implementation 
- **OpenAI (GPT)**: Via official Python SDK
- **Google (Gemini)**: Via google.generativeai package

## Example Output

```
🔹 Sending message to anthropic:
  "What is 512 * 48?"
🤖 Initial Response:
🛠️ Tool Calls (1):
  Call #1:
  Name: calculator
  Arguments: {
    "expression": "512 * 48"
  }
  Result: {
    "result": 24576,
    "expression": "512 * 48"
  }
```

## Contributing

Found a bug? Have a suggestion? PRs and comments welcome!