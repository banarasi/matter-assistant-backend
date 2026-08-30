import json
from typing import Protocol

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPCaller(Protocol):
    async def call(self, tool: str, args: dict) -> dict: ...


class MCPClient:
    def __init__(self, url: str):
        self.url = url

    async def call(self, tool: str, args: dict) -> dict:
        async with streamable_http_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                if getattr(result, "structuredContent", None) is not None:
                    return result.structuredContent
                if not result.content or not hasattr(result.content[0], "text"):
                    raise RuntimeError(f"MCP tool {tool} returned no JSON content")
                payload = json.loads(result.content[0].text)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"MCP tool {tool} returned a non-object response")
                return payload
