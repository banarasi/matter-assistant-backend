import json
from typing import Protocol

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPCaller(Protocol):
    async def call(self, tool: str, args: dict) -> dict: ...


class MCPClient:
    def __init__(self, url: str):
        self.url = url

    async def call(self, tool: str, args: dict) -> dict:
        async with streamablehttp_client(self.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                if getattr(result, "structuredContent", None):
                    return result.structuredContent
                return json.loads(result.content[0].text)
