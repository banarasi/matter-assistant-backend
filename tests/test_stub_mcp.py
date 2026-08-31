"""USE_STUB_MCP: the real Passport tool implementations (rules engine + reference
data from passport_mcp) over an in-memory store, so the wizard can be exercised
end to end — e.g. from the portal — with no MCP server and no Postgres.
"""
from dataclasses import replace

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

import agent_service.app as app_module
from agent_service.app import app, lifespan
from agent_service.config import Settings
from agent_service.graph.builder import build_graph
from agent_service.mcp_client import MCPClient, make_mcp_client
from agent_service.model_client import StubModelClient
from agent_service.stub_mcp import StubMCPClient

from .graph_utils import cards, errors, last_card
from .test_app import parse_sse
from .test_golden import run_happy_path

WHO = {"requested_by": "jane.smith", "correlation_id": "c1"}
CORE = {"matter_name": "Stub matter", "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV",
        "matter_subtype": "MST-POL", "country": "CH", "business_segment": "BS-WM",
        "legal_entity": "LE-UBS-CH", "confidentiality_class": "CC-CONF",
        "pic_employee_id": "E1001", "incident_risk_category": "IRC-HIGH",
        "nfr_taxonomy": "NFR-EP", "risk_theme": "RT-PG"}


def test_use_stub_mcp_from_env(monkeypatch):
    monkeypatch.delenv("USE_STUB_MCP", raising=False)
    assert Settings().use_stub_mcp is False
    monkeypatch.setenv("USE_STUB_MCP", "true")
    assert Settings().use_stub_mcp is True


def test_make_mcp_client_selects_stub(monkeypatch):
    base = Settings()
    assert isinstance(make_mcp_client(replace(base, use_stub_mcp=False)), MCPClient)
    assert isinstance(make_mcp_client(replace(base, use_stub_mcp=True)), StubMCPClient)


async def test_stub_reads_use_real_reference_data():
    stub = StubMCPClient()
    res = await stub.call("get_reference_data", {"domain": "pabu", **WHO})
    assert res["ok"] and any(v["id"] == "PABU-EMP" for v in res["values"])
    res = await stub.call("search_employees", {"query": "jane", **WHO})
    assert res["ok"] and res["employees"][0]["id"] == "E1001"
    res = await stub.call("get_matter", {"matter_id": "MAT-2026-000001", **WHO})
    assert res == {"ok": False, "error": {"code": "MATTER_NOT_FOUND",
                                          "message": "MAT-2026-000001 not found",
                                          "field": None}}


async def test_stub_write_lifecycle_and_idempotency():
    stub = StubMCPClient()
    created = await stub.call("create_matter", {"payload": CORE, "idempotency_key": "k1", **WHO})
    assert created["ok"] and created["matter_id"] == "MAT-2026-001245"
    replay = await stub.call("create_matter", {"payload": CORE, "idempotency_key": "k1", **WHO})
    assert replay == created  # idempotent replay, no second matter
    second = await stub.call("create_matter", {"payload": CORE, "idempotency_key": "k2", **WHO})
    assert second["matter_id"] == "MAT-2026-001246"
    mid = created["matter_id"]

    party = {"matter_id": mid, "org_id": "ORG-BM", "role": "organization", **WHO}
    assert (await stub.call("add_matter_party", {**party, "idempotency_key": "p1"}))["ok"]
    assert (await stub.call("add_matter_party", {**party, "idempotency_key": "p2"}))["ok"]
    budget = {"matter_id": mid, "org_id": "ORG-BM", "amount": 1000, "currency": "USD",
              "fiscal_period": "FY2027", **WHO}
    assert (await stub.call("create_budget", {**budget, "idempotency_key": "b1"}))["ok"]
    dup = await stub.call("create_budget", {**budget, "idempotency_key": "b2"})
    assert dup["error"]["code"] == "BUDGET_EXISTS_FOR_PERIOD"
    alloc = await stub.call("set_cost_allocation", {
        "matter_id": mid, "idempotency_key": "a1", **WHO,
        "allocations": [{"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]})
    assert alloc["ok"] and alloc["allocations"][0]["cc_name"]

    snap = await stub.call("get_matter", {"matter_id": mid, **WHO})
    assert snap["status"] == "draft" and len(snap["parties"]) == 1  # party de-duplicated
    assert snap["budgets"] == [{"org_id": "ORG-BM", "amount": 1000.0, "currency": "USD",
                                "fiscal_period": "FY2027"}]
    assert sum(a["pct"] for a in snap["allocations"]) == 100

    sub = await stub.call("submit_matter", {"matter_id": mid, "idempotency_key": "s1", **WHO})
    assert sub == {"ok": True, "status": "submitted", "matter_id": mid}
    again = await stub.call("submit_matter", {"matter_id": mid, "idempotency_key": "s2", **WHO})
    assert again["error"]["code"] == "INVALID_STATE_TRANSITION"
    assert (await stub.call("get_matter", {"matter_id": mid, **WHO}))["status"] == "submitted"


async def test_stub_unknown_tool_raises():
    with pytest.raises(RuntimeError, match="unknown MCP tool"):
        await StubMCPClient().call("drop_everything", {})


async def test_golden_happy_path_against_stub():
    stub = StubMCPClient()
    graph = build_graph(StubModelClient(), stub, MemorySaver())
    cfg = {"configurable": {"thread_id": "stub-g1"}}
    evs = await run_happy_path(graph, cfg, "stub-g1")
    assert "SubmittedCard" in cards(evs)
    assert "MCP_UNAVAILABLE" not in errors(evs)
    mid = last_card(evs, "SubmittedCard")["matter_id"]
    snap = await stub.call("get_matter", {"matter_id": mid, **WHO})
    assert snap["status"] == "submitted"
    assert len(snap["parties"]) == 2
    assert snap["budgets"][0]["fiscal_period"] == "FY2027"


@pytest.fixture()
async def standalone_app(monkeypatch):
    monkeypatch.setattr(app_module, "settings", replace(
        app_module.settings, checkpointer="memory", use_stub_model=True, use_stub_mcp=True))
    async with lifespan(app):
        yield app


async def test_standalone_app_serves_picklists_without_mcp_server(standalone_app):
    transport = httpx.ASGITransport(app=standalone_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        cid = parse_sse((await c.post("/conversations")).text)[0]["conversation_id"]
        r = await c.post(f"/conversations/{cid}/messages",
                         json={"type": "action", "name": "start"})
        evs = parse_sse(r.text)
        assert not [e for e in evs if e.get("type") == "error"]
        basics = last_card(evs, "BasicInfoCard")
        assert basics["pabu"], "picklists must come from the in-memory stub"
