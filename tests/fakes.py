from passport_mcp import tools


class FakeMCPClient:
    """In-process MCP: dispatches straight to the real tool functions."""

    def __init__(self, ctx: tools.ToolContext):
        self.ctx = ctx

    async def call(self, tool: str, args: dict) -> dict:
        return getattr(tools, tool)(self.ctx, **args)
