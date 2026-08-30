import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.graph.nodes_intake import BasicsExtract
from agent_service.model_client import StubModelClient

from .graph_utils import cards, errors, last_card, send, start

BASICS = {"matter_name": "Employment Investigation - Zurich Office",
          "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV", "matter_subtype": "MST-POL"}
COND = {"business_segment": "BS-WM", "legal_entity": "LE-UBS-CH",
        "confidentiality_class": "CC-CONF"}
PIC = {"pic_employee_id": "E1001", "pic_employee_name": "Jane Smith",
       "incident_risk_category": "IRC-HIGH", "nfr_taxonomy": "NFR-EP",
       "risk_theme": "RT-PG"}


def mkgraph(fake_mcp, extractions=None):
    return build_graph(StubModelClient(extractions=extractions), fake_mcp, MemorySaver())


async def run_happy_path(graph, cfg, cid, country="CH", steps=None):
    """Drive the full wizard to submission. If `steps` is a dict, per-step
    events are recorded into it (currently just the jurisdiction submit) so
    callers can assert on intermediate turns without re-driving the flow."""
    await start(graph, cfg, conversation_id=cid)
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": BASICS})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"country": country}})
    if steps is not None:
        steps["jurisdiction"] = evs
    if country == "CH":
        evs = await send(graph, cfg, {"type": "card_submit", "values": COND})
    await send(graph, cfg, {"type": "card_submit", "values": PIC})
    await send(graph, cfg, {"type": "action", "name": "add_org"})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie",
        "outside_counsel_id": "ORG-BM", "outside_counsel_name": "Baker McKenzie"}})
    await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]}})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": 150000, "currency": "USD", "fiscal_period": "FY2027"}})
    return await send(graph, cfg, {"type": "action", "name": "confirm_submit"})


async def test_golden_happy_path_ch(fake_mcp):
    graph = mkgraph(fake_mcp)
    cfg = {"configurable": {"thread_id": "g1"}}
    evs = await run_happy_path(graph, cfg, "g1")
    assert "SubmittedCard" in cards(evs)
    mid = last_card(evs, "SubmittedCard")["matter_id"]
    snap = await fake_mcp.call("get_matter", {"matter_id": mid})
    assert snap["status"] == "submitted"
    assert len(snap["parties"]) == 2
    assert sum(a["pct"] for a in snap["allocations"]) == 100
    assert snap["budgets"][0]["fiscal_period"] == "FY2027"


async def test_golden_non_ch_skips_cond_fields(fake_mcp):
    graph = mkgraph(fake_mcp)
    cfg = {"configurable": {"thread_id": "g2"}}
    steps = {}
    evs = await run_happy_path(graph, cfg, "g2", country="US", steps=steps)
    assert "SubmittedCard" in cards(evs)
    # make the skip explicit: the US jurisdiction submit never rendered the
    # CH-only conditional-fields card
    assert "AdditionalFieldsCard" not in cards(steps["jurisdiction"])


async def test_golden_allocation_error_recovery(fake_mcp):
    graph = mkgraph(fake_mcp)
    cfg = {"configurable": {"thread_id": "g3"}}
    await start(graph, cfg, conversation_id="g3")
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": BASICS})
    await send(graph, cfg, {"type": "card_submit", "values": {"country": "CH"}})
    await send(graph, cfg, {"type": "card_submit", "values": COND})
    await send(graph, cfg, {"type": "card_submit", "values": PIC})
    await send(graph, cfg, {"type": "action", "name": "add_org"})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 60}, {"cc_id": "100078", "pct": 30}]}})
    assert "ALLOCATION_SUM_INVALID" in errors(evs)
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]}})
    assert "BudgetCard" in cards(evs)


async def test_golden_pic_not_entitled(fake_mcp):
    graph = mkgraph(fake_mcp)
    cfg = {"configurable": {"thread_id": "g4"}}
    await start(graph, cfg, conversation_id="g4")
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": BASICS})
    await send(graph, cfg, {"type": "card_submit", "values": {"country": "US"}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        **PIC, "pic_employee_id": "E1003", "pic_employee_name": "Priya Patel"}})
    assert "PIC_NO_ENTITLEMENT" in errors(evs)


async def test_golden_edit_loop(fake_mcp):
    graph = mkgraph(fake_mcp)
    cfg = {"configurable": {"thread_id": "g5"}}
    await start(graph, cfg, conversation_id="g5")
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": BASICS})
    await send(graph, cfg, {"type": "card_submit", "values": {"country": "CH"}})
    await send(graph, cfg, {"type": "card_submit", "values": COND})
    await send(graph, cfg, {"type": "card_submit", "values": PIC})
    await send(graph, cfg, {"type": "action", "name": "add_org"})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": None}]}})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": 150000, "currency": "USD", "fiscal_period": "FY2027"}})
    evs = await send(graph, cfg, {"type": "action", "name": "edit:allocation"})
    assert "AllocationCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 50}, {"cc_id": "100078", "pct": 50}]}})
    assert "ReviewCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "action", "name": "confirm_submit"})
    assert "SubmittedCard" in cards(evs)
    # the edited 50/50 split (not the original single-cc auto-100) is what
    # actually persisted on the submitted matter
    mid = last_card(evs, "SubmittedCard")["matter_id"]
    snap = await fake_mcp.call("get_matter", {"matter_id": mid})
    assert sorted((a["cc_id"], a["pct"]) for a in snap["allocations"]) == [
        ("100045", 50), ("100078", 50)]


async def test_golden_free_text_offscript(fake_mcp):
    # extraction yields nothing useful; assistant asks for missing fields and re-renders
    graph = mkgraph(fake_mcp, extractions=[BasicsExtract()])
    cfg = {"configurable": {"thread_id": "g6"}}
    await start(graph, cfg, conversation_id="g6")
    await send(graph, cfg, {"type": "action", "name": "start"})
    evs = await send(graph, cfg, {"type": "text", "text": "what does NFR taxonomy mean?"})
    assert any(e["type"] == "text_delta" for e in evs)
    assert "BasicInfoCard" in cards(evs)
