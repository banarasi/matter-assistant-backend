import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient

from .graph_utils import cards, errors, last_card, send, start


async def drive_to_pic_risk(graph, cfg, cid):
    await start(graph, cfg, conversation_id=cid)
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "matter_name": "Employment Investigation - Zurich Office",
        "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV", "matter_subtype": "MST-POL"}})
    await send(graph, cfg, {"type": "card_submit", "values": {"country": "CH"}})
    return await send(graph, cfg, {"type": "card_submit", "values": {
        "business_segment": "BS-WM", "legal_entity": "LE-UBS-CH",
        "confidentiality_class": "CC-CONF"}})


RISK = {"incident_risk_category": "IRC-HIGH", "nfr_taxonomy": "NFR-EP", "risk_theme": "RT-PG"}


@pytest.fixture()
def graph(fake_mcp):
    return build_graph(StubModelClient(), fake_mcp, MemorySaver())


async def test_pic_search_and_verify(graph):
    cfg = {"configurable": {"thread_id": "r1"}}
    evs = await drive_to_pic_risk(graph, cfg, "r1")
    assert "PicRiskCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "action", "name": "search_employee", "query": "jane"})
    assert last_card(evs, "PicRiskCard")["employees"][0]["id"] == "E1001"
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1001", "pic_employee_name": "Jane Smith", **RISK}})
    names = cards(evs)
    assert "AccessVerifiedBadge" in names and "MatterCreatedCard" in names
    assert last_card(evs, "MatterCreatedCard")["matter_id"] == "MAT-2026-001245"


async def test_pic_not_entitled_blocks(graph):
    cfg = {"configurable": {"thread_id": "r2"}}
    await drive_to_pic_risk(graph, cfg, "r2")
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1003", "pic_employee_name": "Priya Patel", **RISK}})
    assert "PIC_NO_ENTITLEMENT" in errors(evs)
    assert "MatterCreatedCard" not in cards(evs)
    # recover with an entitled PIC
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1001", "pic_employee_name": "Jane Smith", **RISK}})
    assert "MatterCreatedCard" in cards(evs)


async def test_create_shell_is_idempotent(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "r3"}}
    await drive_to_pic_risk(graph, cfg, "r3")
    await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1001", "pic_employee_name": "Jane Smith", **RISK}})
    # only one matter row exists even though the node ran once and could be retried
    res = await fake_mcp.call("get_matter", {"matter_id": "MAT-2026-001246"})
    assert res["ok"] is False  # no second matter was created
