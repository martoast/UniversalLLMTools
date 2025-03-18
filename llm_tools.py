#!/usr/bin/env python3
"""
Simple LLM calculator tool implementation.
"""
import json
import re
import requests
from enum import Enum
from typing import Dict, List, Any, Tuple

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"

class Tool:
    def __init__(self, name: str, description: str, parameters: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_provider_format(self, provider: Provider) -> Dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                p["name"]: {
                    "type": p["type"],
                    "description": p["description"]
                } for p in self.parameters
            },
            "required": [p["name"] for p in self.parameters if p.get("required", False)]
        }

        if provider == Provider.ANTHROPIC:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": schema
            }
        elif provider == Provider.GOOGLE:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": schema
            }
        elif provider == Provider.OPENAI:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": schema
                }
            }

class ToolCall:
    def __init__(self, id: str, name: str, arguments: Dict[str, Any]):
        self.id = id
        self.name = name
        self.arguments = arguments

class ToolResult:
    def __init__(self, id: str, name: str, output: Dict[str, Any]):
        self.id = id
        self.name = name
        self.output = output

class LLMToolClient:
    def __init__(self, provider: Provider, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self._setup_client()

    def _setup_client(self):
        if self.provider == Provider.ANTHROPIC:
            self.api_url = "https://api.anthropic.com/v1/messages"
            self.headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        elif self.provider == Provider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == Provider.GOOGLE:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai

    def call_with_tools(self, message: str, tools: List[Tool], **kwargs) -> Tuple[Any, List[ToolCall], str]:
        provider_tools = [tool.to_provider_format(self.provider) for tool in tools]
        system = "You have access to a calculator tool. You MUST use it for ANY mathematical calculations requested."

        if self.provider == Provider.ANTHROPIC:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": kwargs.get("model", "claude-3-sonnet-20240229"),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "messages": [{"role": "user", "content": message}],
                    "system": system,
                    "tools": provider_tools
                }
            ).json()

        elif self.provider == Provider.OPENAI:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
            response = self.client.chat.completions.create(
                model=kwargs.get("model", "gpt-4"),
                messages=messages,
                tools=provider_tools,
                tool_choice="auto"
            )

        elif self.provider == Provider.GOOGLE:
            model = self.client.GenerativeModel(model_name=kwargs.get("model", "gemini-pro"))
            contents = [
                {"role": "user", "parts": [{"text": system}]},
                {"role": "model", "parts": [{"text": "Understood."}]},
                {"role": "user", "parts": [{"text": message}]}
            ]
            response = model.generate_content(
                contents=contents,
                tools={"function_declarations": provider_tools}
            )

        return response, self._parse_tool_calls(response), self._extract_text(response)

    def _parse_tool_calls(self, response: Any) -> List[ToolCall]:
        tool_calls = []
        
        if self.provider == Provider.ANTHROPIC:
            try:
                for content_item in response.get("content", []):
                    if content_item.get("type") == "tool_use":
                        tool_calls.append(ToolCall(
                            id=content_item.get("id", f"tool_{len(tool_calls)}"),
                            name=content_item.get("name", "unknown"),
                            arguments=content_item.get("input", {})
                        ))
            except:
                pass
                
            if not tool_calls:
                text = self._extract_text(response)
                tool_calls = self._parse_tool_calls_from_text(text)
                
            return tool_calls

        elif self.provider == Provider.OPENAI:
            for choice in response.choices:
                if hasattr(choice.message, "tool_calls"):
                    for call in choice.message.tool_calls:
                        tool_calls.append(ToolCall(
                            id=call.id,
                            name=call.function.name,
                            arguments=json.loads(call.function.arguments)
                        ))

        elif self.provider == Provider.GOOGLE:
            if hasattr(response, "candidates"):
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            # Convert MapComposite to dict
                            args = {}
                            if hasattr(part.function_call.args, "items"):
                                for key, value in part.function_call.args.items():
                                    args[key] = str(value)  # Convert all values to strings
                            else:
                                args = {"expression": str(part.function_call.args)}
                                
                            tool_calls.append(ToolCall(
                                id=f"call_{len(tool_calls)}",
                                name=part.function_call.name,
                                arguments=args
                            ))

        return tool_calls

    def _parse_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        tool_calls = []
        calculator_match = re.search(r'calculator.+?(\d+\s*[\+\-\*\/]\s*\d+)', text, re.IGNORECASE)
        if calculator_match:
            expression = calculator_match.group(1).strip()
            tool_calls.append(ToolCall(
                id="tool_calculator",
                name="calculator",
                arguments={"expression": expression}
            ))
        return tool_calls

    def _extract_text(self, response: Any) -> str:
        if self.provider == Provider.ANTHROPIC:
            text_parts = []
            for item in response.get("content", []):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)
        elif self.provider == Provider.OPENAI:
            return response.choices[0].message.content or ""
        elif self.provider == Provider.GOOGLE:
            return response.text if hasattr(response, "text") else ""
        return ""

def execute_calculator(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}