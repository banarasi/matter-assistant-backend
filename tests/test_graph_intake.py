import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient
from agent_service.graph.nodes_intake import BasicsExtract, match_label

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


async def test_jurisdiction_surfaces_authoritative_invalid_country(graph):
    await start(graph, CFG)
    await send(graph, CFG, {"type": "action", "name": "start"})
    await send(graph, CFG, {"type": "card_submit", "values": {
        "matter_name": "X", "pabu": "PABU-EMP", "matter_type": "MT-EMP-INV",
        "matter_subtype": "MST-POL"}})
    evs = await send(graph, CFG, {"type": "card_submit", "values": {"country": "XX"}})
    assert "INVALID_PICKLIST_VALUE" in errors(evs)
    assert "JurisdictionCard" in cards(evs)


def test_label_matching_does_not_guess_ambiguous_or_blank_values():
    values = [{"id": "1", "label": "Legal One"}, {"id": "2", "label": "Legal Two"}]
    assert match_label(values, "legal") is None
    assert match_label(values, "   ") is None
    assert match_label(values, "Legal One") == "1"


class CountingStubModelClient(StubModelClient):
    """Counts extract() calls to prove interrupt-resume replay never re-runs LLM work."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extract_calls = 0

    async def extract(self, schema, instruction, text):
        self.extract_calls += 1
        return await super().extract(schema, instruction, text)


async def test_no_duplicate_llm_calls_across_turns(fake_mcp):
    stub = CountingStubModelClient(extractions=[BasicsExtract(
        matter_name="Employment Investigation - Zurich Office",
        pabu_label="Employment Legal",
        matter_type_label="Employment Investigation",
        matter_subtype_label="Policy Violation")])
    graph = build_graph(stub, fake_mcp, MemorySaver())
    cfg = {"configurable": {"thread_id": "t-count"}}
    await start(graph, cfg, conversation_id="t-count")
    await send(graph, cfg, {"type": "action", "name": "start"})
    await send(graph, cfg, {"type": "text",
                            "text": "employment investigation, policy violation, Zurich"})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {}})
    assert "JurisdictionCard" in cards(evs)
    # The card_submit resume replays the basics node body, but the single-interrupt
    # pattern means the text-turn's extract() is never re-executed.
    assert stub.extract_calls == 1


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


class FailingConversationModel(StubModelClient):
    async def converse_stream(self, system, user):
        raise RuntimeError("provider unavailable")
        yield  # pragma: no cover - makes this an async generator


async def test_model_failure_keeps_checkpoint_and_renders_form(fake_mcp):
    graph = build_graph(FailingConversationModel(), fake_mcp, MemorySaver())
    cfg = {"configurable": {"thread_id": "t-model-failure"}}
    await start(graph, cfg, conversation_id="t-model-failure")
    await send(graph, cfg, {"type": "action", "name": "start"})
    evs = await send(graph, cfg, {"type": "text", "text": "help me"})
    assert "MODEL_UNAVAILABLE" in errors(evs)
    assert "BasicInfoCard" in cards(evs)


class FailingExtractionModel(StubModelClient):
    async def extract(self, schema, system, user):
        raise RuntimeError("provider secret detail")


async def test_extraction_failure_is_retryable_and_does_not_leak_details(fake_mcp):
    graph = build_graph(FailingExtractionModel(), fake_mcp, MemorySaver())
    cfg = {"configurable": {"thread_id": "t-extraction-failure"}}
    await start(graph, cfg, conversation_id="t-extraction-failure")
    await send(graph, cfg, {"type": "action", "name": "start"})
    evs = await send(graph, cfg, {"type": "text", "text": "help me"})
    messages = [event.get("message", "") for event in evs if event.get("type") == "error"]
    assert "MODEL_UNAVAILABLE" in errors(evs)
    assert all("secret detail" not in message for message in messages)
    assert "BasicInfoCard" in cards(evs)
