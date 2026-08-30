import asyncio
import json
from types import SimpleNamespace

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
    assert len(events[0]["conversation_id"]) == 32
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
    assert r.headers["cache-control"] == "no-store"


class ExplodingGraph:
    async def aget_state(self, *args, **kwargs):
        interrupt = SimpleNamespace(value={"type": "card", "card": "WelcomeCard", "props": {}})
        return SimpleNamespace(values={"current_stage": "welcome"},
                               tasks=[SimpleNamespace(interrupts=[interrupt])])

    def astream(self, *args, **kwargs):
        async def gen():
            yield {"type": "stage_change", "stage": "welcome"}
            raise RuntimeError("boom")
        return gen()


class BlockingGraph(ExplodingGraph):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def astream(self, *args, **kwargs):
        async def gen():
            self.started.set()
            await self.release.wait()
            yield {"type": "stage_change", "stage": "welcome"}
        return gen()


async def test_stream_error_still_emits_done(client):
    app.state.graph_override = ExplodingGraph()
    r = await client.post(f"/conversations/{'a' * 32}/messages",
                          json={"type": "action", "name": "start"})
    events = parse_sse(r.text)
    assert any(e.get("type") == "error" and e.get("code") == "INTERNAL" for e in events)
    assert all("boom" not in e.get("message", "") for e in events)
    assert events[-1]["type"] == "done"


async def test_security_headers_apply_to_api_responses(client):
    response = await client.get("/healthz")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


async def test_concurrent_resume_of_same_conversation_is_rejected(client):
    graph = BlockingGraph()
    app.state.graph_override = graph
    cid = "c" * 32
    first = asyncio.create_task(client.post(
        f"/conversations/{cid}/messages", json={"type": "action", "name": "start"}))
    await asyncio.wait_for(graph.started.wait(), timeout=2)
    second = await client.post(
        f"/conversations/{cid}/messages", json={"type": "action", "name": "start"})
    assert second.status_code == 409
    graph.release.set()
    assert (await first).status_code == 200
    assert cid not in app.state.active_conversations


async def test_unknown_conversation_returns_404(client):
    unknown = "0" * 32
    assert (await client.get(f"/conversations/{unknown}/state")).status_code == 404
    response = await client.post(
        f"/conversations/{unknown}/messages", json={"type": "action", "name": "start"})
    assert response.status_code == 404


async def test_malformed_conversation_id_is_rejected(client):
    assert (await client.get("/conversations/not-an-id/state")).status_code == 422


async def test_message_payload_is_validated_before_graph_resume(client):
    response = await client.post(
        f"/conversations/{'b' * 32}/messages",
        json={"type": "card_submit"},
    )
    assert response.status_code == 422


async def test_blank_user_identity_is_rejected(client):
    response = await client.post("/conversations", headers={"X-User-Id": "   "})
    assert response.status_code == 400
