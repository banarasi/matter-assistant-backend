import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient
from agent_service.graph.nodes_intake import BasicsExtract

from .graph_utils import cards, errors, last_card, send, start

CFG = {"configurable": {"thread_id": "t1"}}


@pytest.fixture()
def graph(fake_mcp):
    return build_graph(StubModelClient(), fake_mcp, MemorySaver())


async def test_welcome_then_basics(graph):
    evs = await start(graph, CFG)
    assert cards(evs) == ["WelcomeCard"]
    evs = await send(graph, CFG, {"type": "action", "name": "start"})
    assert "BasicInfoCard" in cards(evs)
    props = last_card(evs, "BasicInfoCard")
    assert any(v["id"] == "PABU-EMP" for v in props["pabu"])


async def test_basics_validation_and_advance(graph):
    await start(graph, CFG)
    await send(graph, CFG, {"type": "action", "name": "start"})
    evs = await send(graph, CFG, {"type": "card_submit", "values": {"matter_name": "X"}})
    assert "REQUIRED_FIELD_MISSING" in errors(evs)
    evs = await send(graph, CFG, {"type": "card_submit", "values": {
        "matter_name": "Employment Investigation - Zurich Office",
        "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV", "matter_subtype": "MST-POL"}})
    assert "JurisdictionCard" in cards(evs)


async def test_jurisdiction_ch_branches_to_cond_fields(graph):
    await start(graph, CFG)
    await send(graph, CFG, {"type": "action", "name": "start"})
    await send(graph, CFG, {"type": "card_submit", "values": {
        "matter_name": "X", "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV",
        "matter_subtype": "MST-POL"}})
    evs = await send(graph, CFG, {"type": "card_submit", "values": {"country": "CH"}})
    assert "AdditionalFieldsCard" in cards(evs)
    assert last_card(evs, "AdditionalFieldsCard")["required"] == [
        "business_segment", "legal_entity", "confidentiality_class"]


async def test_jurisdiction_us_skips_cond_fields(fake_mcp):
    graph = build_graph(StubModelClient(), fake_mcp, MemorySaver())
    cfg = {"configurable": {"thread_id": "t-us"}}
    await start(graph, cfg, conversation_id="t-us")
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "card_submit", "values": {
        "matter_name": "X", "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV",
        "matter_subtype": "MST-POL"}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"country": "US"}})
    # pic_risk node doesn't exist yet -> graph stops after jurisdiction; assert no
    # AdditionalFieldsCard was emitted
    assert "AdditionalFieldsCard" not in cards(evs)


async def test_free_text_extraction_prefills(fake_mcp):
    stub = StubModelClient(extractions=[BasicsExtract(
        matter_name="Employment Investigation - Zurich Office",
        pabu_label="Employment Legal",
        matter_type_label="Employment Investigation",
        matter_subtype_label="Policy Violation")])
    graph = build_graph(stub, fake_mcp, MemorySaver())
    cfg = {"configurable": {"thread_id": "t-x"}}
    await start(graph, cfg, conversation_id="t-x")
    await send(graph, cfg, {"type": "action", "name": "start"})
    evs = await send(graph, cfg, {"type": "text",
                                  "text": "employment investigation, policy violation, Zurich"})
    props = last_card(evs, "BasicInfoCard")
    assert props["values"]["pabu"] == "PABU-EMP"
    assert props["values"]["matter_subtype"] == "MST-POL"
    assert props["missing"] == []
