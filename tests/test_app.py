import json

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.app import app
from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient


def parse_sse(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):])
            for line in text.splitlines() if line.startswith("data: ")]


@pytest.fixture()
async def client(fake_mcp):
    app.state.graph_override = build_graph(StubModelClient(), fake_mcp, MemorySaver())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    del app.state.graph_override


async def test_conversation_start_streams_welcome(client):
    r = await client.post("/conversations")
    events = parse_sse(r.text)
    assert events[0]["type"] == "conversation"
    assert any(e.get("card") == "WelcomeCard" for e in events)
    assert events[-1]["type"] == "done"


async def test_message_roundtrip_and_state(client):
    r = await client.post("/conversations")
    cid = parse_sse(r.text)[0]["conversation_id"]
    r = await client.post(f"/conversations/{cid}/messages",
                          json={"type": "action", "name": "start"})
    events = parse_sse(r.text)
    assert any(e.get("card") == "BasicInfoCard" for e in events)
    r = await client.get(f"/conversations/{cid}/state")
    body = r.json()
    assert body["values"]["current_stage"] == "basics"
    assert body["pending_card"]["card"] == "BasicInfoCard"


class ExplodingGraph:
    def astream(self, *args, **kwargs):
        async def gen():
            yield {"type": "stage_change", "stage": "welcome"}
            raise RuntimeError("boom")
        return gen()


async def test_stream_error_still_emits_done(client):
    app.state.graph_override = ExplodingGraph()
    r = await client.post("/conversations/abc123/messages",
                          json={"type": "action", "name": "start"})
    events = parse_sse(r.text)
    assert any(e.get("type") == "error" and e.get("code") == "INTERNAL" for e in events)
    assert events[-1]["type"] == "done"
