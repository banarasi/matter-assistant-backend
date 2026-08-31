import json
from typing import Protocol

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import Settings


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


def make_mcp_client(settings: Settings) -> MCPCaller:
    """Real Streamable-HTTP client, or the in-process stub when USE_STUB_MCP=true."""
    if settings.use_stub_mcp:
        from .stub_mcp import StubMCPClient  # lazy: pulls in passport_mcp

        return StubMCPClient()
    return MCPClient(settings.mcp_server_url)
