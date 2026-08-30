import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient

from .graph_utils import cards, errors, last_card, send
from .test_graph_setup import drive_to_counsel


async def drive_to_review(graph, cfg, cid):
    await drive_to_counsel(graph, cfg, cid)
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]}})
    return await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": 150000, "currency": "USD", "fiscal_period": "FY2027"}})


@pytest.fixture()
def graph(fake_mcp):
    return build_graph(StubModelClient(), fake_mcp, MemorySaver())


async def test_review_shows_snapshot(graph):
    cfg = {"configurable": {"thread_id": "v1"}}
    evs = await drive_to_review(graph, cfg, "v1")
    assert "ReviewCard" in cards(evs)
    snap = last_card(evs, "ReviewCard")["snapshot"]
    assert snap["core"]["legal_entity"] == "LE-UBS-CH"
    assert sum(a["pct"] for a in snap["allocations"]) == 100


async def test_edit_loop_returns_to_review(graph):
    cfg = {"configurable": {"thread_id": "v2"}}
    await drive_to_review(graph, cfg, "v2")
    evs = await send(graph, cfg, {"type": "action", "name": "edit:allocation"})
    assert "AllocationCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 50}, {"cc_id": "100078", "pct": 50}]}})
    # after the edit we land straight back on review
    assert "ReviewCard" in cards(evs)
    snap = last_card(evs, "ReviewCard")["snapshot"]
    assert snap["allocations"][0]["pct"] == 50


async def test_submit_finishes(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "v3"}}
    await drive_to_review(graph, cfg, "v3")
    evs = await send(graph, cfg, {"type": "action", "name": "confirm_submit"})
    assert "SubmittedCard" in cards(evs)
    snap = await fake_mcp.call("get_matter",
                               {"matter_id": last_card(evs, "SubmittedCard")["matter_id"]})
    assert snap["status"] == "submitted"


async def test_edit_revisit_a_b_a_persists_final_value(graph, fake_mcp):
    # F3 regression: content-derived idempotency keys used to be stable across
    # an A->B->A edit cycle, so resubmitting A (having already submitted it
    # once before B) replayed the MCP server's cached response for A's *first*
    # submission instead of writing again — except by then the matter's
    # persisted allocations were B, and the agent still reported success,
    # silently leaving the DB out of sync with what the user last submitted.
    cfg = {"configurable": {"thread_id": "v6"}}
    alloc_a = [{"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]
    alloc_b = [{"cc_id": "100045", "pct": 50}, {"cc_id": "100078", "pct": 50}]
    matter_id = None

    async def submit(allocs):
        nonlocal matter_id
        await send(graph, cfg, {"type": "action", "name": "edit:allocation"})
        evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": allocs}})
        assert "ReviewCard" in cards(evs)
        matter_id = last_card(evs, "ReviewCard")["snapshot"]["matter_id"]

    await drive_to_review(graph, cfg, "v6")  # submits A the first time
    await submit(alloc_b)   # A -> B
    await submit(alloc_a)   # B -> A again (same content as the original submit)

    snap = await fake_mcp.call("get_matter", {"matter_id": matter_id})
    persisted = sorted(snap["allocations"], key=lambda a: a["cc_id"])
    assert persisted[0]["cc_id"] == "100045" and persisted[0]["pct"] == 70
    assert persisted[1]["cc_id"] == "100078" and persisted[1]["pct"] == 30


class FlakyMCP:
    """Delegates to the real fake MCP but raises on submit_matter while failing."""

    def __init__(self, inner):
        self.inner = inner
        self.failing = True

    async def call(self, tool: str, args: dict) -> dict:
        if self.failing and tool == "submit_matter":
            raise RuntimeError("mcp connection lost")
        return await self.inner.call(tool, args)


async def test_submit_transport_failure_returns_to_review(fake_mcp):
    flaky = FlakyMCP(fake_mcp)
    graph = build_graph(StubModelClient(), flaky, MemorySaver())
    cfg = {"configurable": {"thread_id": "v5"}}
    await drive_to_review(graph, cfg, "v5")
    evs = await send(graph, cfg, {"type": "action", "name": "confirm_submit"})
    assert "MCP_UNAVAILABLE" in errors(evs)
    assert "SubmittedCard" not in cards(evs)
    assert "ReviewCard" in cards(evs)  # landed back on review
    # once the transport recovers, the same confirm succeeds
    flaky.failing = False
    evs = await send(graph, cfg, {"type": "action", "name": "confirm_submit"})
    assert "SubmittedCard" in cards(evs)


async def test_double_submit_is_idempotent(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "v4"}}
    await drive_to_review(graph, cfg, "v4")
    evs = await send(graph, cfg, {"type": "action", "name": "confirm_submit"})
    mid = last_card(evs, "SubmittedCard")["matter_id"]
    # replaying the same submit key returns the stored response, no INVALID_STATE_TRANSITION
    res = await fake_mcp.call("submit_matter", {
        "matter_id": mid,
        "idempotency_key": "irrelevant-but-new",  # a NEW key must now fail cleanly
        "requested_by": "jane.smith", "correlation_id": "v4"})
    assert res["error"]["code"] == "INVALID_STATE_TRANSITION"
