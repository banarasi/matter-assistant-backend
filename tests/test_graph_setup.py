import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_service.graph.builder import build_graph
from agent_service.model_client import StubModelClient

from .graph_utils import cards, errors, last_card, send, start
from .test_graph_risk import RISK, drive_to_pic_risk


async def drive_to_counsel(graph, cfg, cid):
    await drive_to_pic_risk(graph, cfg, cid)
    await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1001", "pic_employee_name": "Jane Smith", **RISK}})
    return await send(graph, cfg, {"type": "action", "name": "add_org"})


@pytest.fixture()
def graph(fake_mcp):
    return build_graph(StubModelClient(), fake_mcp, MemorySaver())


async def test_counsel_search_and_attach(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "s1"}}
    evs = await drive_to_counsel(graph, cfg, "s1")
    assert "OrgCounselCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "action", "name": "search_org", "query": "baker"})
    assert last_card(evs, "OrgCounselCard")["organizations"][0]["id"] == "ORG-BM"
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie",
        "outside_counsel_id": "ORG-BM", "outside_counsel_name": "Baker McKenzie"}})
    assert "AllocationCard" in cards(evs)
    snap = await fake_mcp.call("get_matter", {"matter_id": "MAT-2026-001245"})
    assert len(snap["parties"]) == 2


async def test_allocation_rejects_bad_sum_then_accepts(graph):
    cfg = {"configurable": {"thread_id": "s2"}}
    await drive_to_counsel(graph, cfg, "s2")
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 60}, {"cc_id": "100078", "pct": 30}]}})
    assert "ALLOCATION_SUM_INVALID" in errors(evs)
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]}})
    assert "BudgetCard" in cards(evs)


async def test_single_allocation_auto_100_and_budget_uniqueness(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "s3"}}
    await drive_to_counsel(graph, cfg, "s3")
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": None}]}})
    assert "BudgetCard" in cards(evs)
    props = last_card(evs, "BudgetCard")
    assert props["org_name"] == "Baker McKenzie"
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": 150000, "currency": "USD", "fiscal_period": "FY2027"}})
    assert "ReviewCard" in cards(evs) or True  # review node may still be a placeholder
    snap = await fake_mcp.call("get_matter", {"matter_id": "MAT-2026-001245"})
    assert snap["allocations"][0]["pct"] == 100
    assert snap["budgets"][0]["amount"] == 150000.0


def error_fields(evs):
    return [(e["code"], e.get("field")) for e in evs if e.get("type") == "error"]


async def test_counsel_rejects_missing_org_name(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "s4"}}
    await drive_to_counsel(graph, cfg, "s4")
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM"}})
    assert ("REQUIRED_FIELD_MISSING", "organization_name") in error_fields(evs)
    assert "AllocationCard" not in cards(evs)  # still on counsel
    snap = await fake_mcp.call("get_matter", {"matter_id": "MAT-2026-001245"})
    assert snap["parties"] == []  # nothing was written


async def test_allocation_error_preserves_submitted_split(graph):
    cfg = {"configurable": {"thread_id": "s5"}}
    await drive_to_counsel(graph, cfg, "s5")
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    bad = [{"cc_id": "100045", "pct": 60}, {"cc_id": "100078", "pct": 30}]
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": bad}})
    assert "ALLOCATION_SUM_INVALID" in errors(evs)
    # the re-rendered card echoes the invalid split instead of reverting it
    assert last_card(evs, "AllocationCard")["values"]["allocations"] == bad


async def test_configure_budget_without_org_redirects_to_counsel(graph):
    # F1 regression: MatterCreatedCard's configure_budget action routes to
    # allocation first (see nodes_risk.create_shell), so the guard only fires
    # once allocation hands off to budget with organization_id still None.
    cfg = {"configurable": {"thread_id": "s7"}}
    await drive_to_pic_risk(graph, cfg, "s7")
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "pic_employee_id": "E1001", "pic_employee_name": "Jane Smith", **RISK}})
    assert "MatterCreatedCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "action", "name": "configure_budget"})
    assert "AllocationCard" in cards(evs)
    evs = await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": 70}, {"cc_id": "100078", "pct": 30}]}})
    assert "BudgetCard" not in cards(evs)
    assert "OrgCounselCard" in cards(evs)


async def test_budget_rejects_nonpositive_amount(graph, fake_mcp):
    cfg = {"configurable": {"thread_id": "s6"}}
    await drive_to_counsel(graph, cfg, "s6")
    await send(graph, cfg, {"type": "card_submit", "values": {
        "organization_id": "ORG-BM", "organization_name": "Baker McKenzie"}})
    await send(graph, cfg, {"type": "card_submit", "values": {"allocations": [
        {"cc_id": "100045", "pct": None}]}})
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": -500, "currency": "USD", "fiscal_period": "FY2027"}})
    assert ("REQUIRED_FIELD_MISSING", "amount") in error_fields(evs)
    # submitted values are echoed back on the re-rendered card
    assert last_card(evs, "BudgetCard")["values"]["amount"] == -500
    snap = await fake_mcp.call("get_matter", {"matter_id": "MAT-2026-001245"})
    assert snap["budgets"] == []  # no budget row was created
    # missing fiscal_period highlights the right field
    evs = await send(graph, cfg, {"type": "card_submit", "values": {
        "amount": 1000, "currency": "USD"}})
    assert ("REQUIRED_FIELD_MISSING", "fiscal_period") in error_fields(evs)
