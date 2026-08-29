async def test_fake_mcp_roundtrip(fake_mcp):
    res = await fake_mcp.call("search_employees", {"query": "jane"})
    assert res["ok"] and res["employees"][0]["id"] == "E1001"


def test_real_client_importable():
    from agent_service.mcp_client import MCPClient
    assert MCPClient("http://localhost:8081/mcp").url
