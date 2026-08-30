from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from .. import events, validation
from ..mcp_client import MCPCaller
from ..state import MatterDraft
from .nodes_intake import fetch_values, idem_key, nxt

WHO = lambda s: {"requested_by": s.requested_by, "correlation_id": s.correlation_id}

COUNSEL_FIELDS = ("organization_id", "organization_name",
                   "outside_counsel_id", "outside_counsel_name")


async def _call(mcp: MCPCaller, writer, tool: str, args: dict) -> dict | None:
    # Same guard shape as nodes_risk._call / nodes_intake.fetch_values: a
    # transport failure (MCP server down/unreachable) is an agent-side concern,
    # not a business error returned by a tool, so it gets its own
    # MCP_UNAVAILABLE event and the caller treats it as a failed call (None)
    # rather than crashing the node.
    try:
        return await mcp.call(tool, args)
    except Exception as e:
        writer(events.error("MCP_UNAVAILABLE", f"{tool} unavailable: {e}", None))
        return None


def make_setup_nodes(model, mcp: MCPCaller):

    async def counsel(state: MatterDraft) -> Command:
        # Single-interrupt-per-execution pattern (see nodes_intake.py header
        # comment / nodes_risk.pic_risk): search results live transiently in
        # state.ui_results (cleared on hand-off to allocation) rather than in a
        # node-local loop, since the node body only runs once per turn.
        writer = get_stream_writer()
        writer(events.stage("counsel"))
        values = {f: getattr(state, f) for f in COUNSEL_FIELDS}
        c = events.card(
            "OrgCounselCard",
            organizations=state.ui_results.get("organizations", []),
            counsel=state.ui_results.get("counsel", []),
            # each key is only present in ui_results after that section's
            # search action ran (possibly as []), so these distinguish
            # "searched, no matches" from "never searched" per section.
            org_searched="organizations" in state.ui_results,
            counsel_searched="counsel" in state.ui_results,
            values=values,
        )
        writer(c)
        payload = interrupt(c)
        ptype = payload.get("type")

        if ptype == "action" and payload.get("name") == "search_org":
            res = await _call(mcp, writer, "search_organizations",
                              {"query": payload.get("query", "")})
            if res is None:
                return Command(goto="counsel")
            return Command(
                update={"ui_results": {**state.ui_results, "organizations": res.get("items", [])}},
                goto="counsel")

        if ptype == "action" and payload.get("name") == "search_counsel":
            res = await _call(mcp, writer, "search_outside_counsel",
                              {"query": payload.get("query", "")})
            if res is None:
                return Command(goto="counsel")
            return Command(
                update={"ui_results": {**state.ui_results, "counsel": res.get("items", [])}},
                goto="counsel")

        if ptype == "card_submit":
            updates = {k: v for k, v in payload.get("values", {}).items()
                      if k in COUNSEL_FIELDS}
            merged = {**values, **updates}
            if not merged["organization_id"] or not merged["organization_name"]:
                field = ("organization_id" if not merged["organization_id"]
                         else "organization_name")
                writer(events.error("REQUIRED_FIELD_MISSING",
                                    "select an organization", field))
                return Command(update=updates, goto="counsel")
            res = await _call(mcp, writer, "add_matter_party", {
                "matter_id": state.matter_id, "org_id": merged["organization_id"],
                "role": "organization",
                "idempotency_key": idem_key(state, "party_org",
                                            {"org": merged["organization_id"]}),
                **WHO(state)})
            if res is None:
                return Command(update=updates, goto="counsel")
            if not res.get("ok"):
                e = res["error"]
                writer(events.error(e["code"], e["message"], e.get("field")))
                return Command(update=updates, goto="counsel")
            if merged["outside_counsel_id"]:
                res2 = await _call(mcp, writer, "add_matter_party", {
                    "matter_id": state.matter_id, "org_id": merged["outside_counsel_id"],
                    "role": "outside_counsel",
                    "idempotency_key": idem_key(state, "party_oc",
                                                {"oc": merged["outside_counsel_id"]}),
                    **WHO(state)})
                if res2 is None:
                    return Command(update=updates, goto="counsel")
                if not res2.get("ok"):
                    e = res2["error"]
                    writer(events.error(e["code"], e["message"], e.get("field")))
                    return Command(update=updates, goto="counsel")
            dest = nxt(state, "allocation")
            return Command(
                update={**updates, "current_stage": dest, "return_to": None, "ui_results": {}},
                goto=dest)

        # unknown payload type -> re-render
        return Command(goto="counsel")

    async def allocation(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("allocation"))
        res = await _call(mcp, writer, "search_cost_centers", {"query": ""})
        cost_centers = res.get("items", []) if res is not None else []
        c = events.card("AllocationCard", cost_centers=cost_centers,
                        values={"allocations": state.allocations})
        writer(c)
        payload = interrupt(c)
        if payload.get("type") != "card_submit":
            return Command(goto="allocation")
        allocs = payload["values"].get("allocations", [])
        if len(allocs) > 1 and validation.allocations_total(allocs) != 100:
            writer(events.error("ALLOCATION_SUM_INVALID",
                                "allocations must total exactly 100%", "allocations"))
            # keep the submitted (invalid) split so the re-rendered card shows
            # the user's input instead of silently reverting it
            return Command(update={"allocations": allocs}, goto="allocation")
        res = await _call(mcp, writer, "set_cost_allocation", {
            "matter_id": state.matter_id, "allocations": allocs,
            "idempotency_key": idem_key(state, "alloc", {"a": allocs}),
            **WHO(state)})
        if res is None:
            return Command(update={"allocations": allocs}, goto="allocation")
        if not res.get("ok"):
            e = res["error"]
            writer(events.error(e["code"], e["message"], e.get("field")))
            return Command(update={"allocations": allocs}, goto="allocation")
        dest = nxt(state, "budget")
        return Command(
            update={"allocations": res["allocations"], "current_stage": dest,
                    "return_to": None},
            goto=dest)

    async def budget(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("budget"))
        fiscal_periods = await fetch_values(mcp, "fiscal_period")
        c = events.card(
            "BudgetCard", org_name=state.organization_name,
            fiscal_periods=fiscal_periods,
            # transient echo of the last (rejected) submission so the user's
            # input re-renders after a validation-error goto-self loop
            values=state.ui_results.get("budget_draft")
                   or {"amount": None, "currency": "USD", "fiscal_period": None},
        )
        writer(c)
        payload = interrupt(c)
        if payload.get("type") != "card_submit":
            return Command(goto="budget")
        v = payload["values"]

        def echo():  # preserve the submitted values across the goto-self loop
            return Command(update={"ui_results": {**state.ui_results, "budget_draft": v}},
                           goto="budget")

        try:
            amount = float(v.get("amount") or 0)
        except (ValueError, TypeError):
            amount = 0.0
        if amount <= 0:
            writer(events.error("REQUIRED_FIELD_MISSING",
                                "a positive amount is required", "amount"))
            return echo()
        if not v.get("fiscal_period"):
            writer(events.error("REQUIRED_FIELD_MISSING",
                                "fiscal period is required", "fiscal_period"))
            return echo()
        res = await _call(mcp, writer, "create_budget", {
            "matter_id": state.matter_id, "org_id": state.organization_id,
            "amount": amount, "currency": v.get("currency", "USD"),
            "fiscal_period": v["fiscal_period"],
            "idempotency_key": idem_key(state, "budget", v),
            **WHO(state)})
        if res is None:
            return echo()
        if not res.get("ok"):
            e = res["error"]
            writer(events.error(e["code"], e["message"], e.get("field")))
            return echo()
        new_budgets = state.budgets + [{
            "org_id": state.organization_id, "org_name": state.organization_name,
            "amount": amount, "currency": v.get("currency", "USD"),
            "fiscal_period": v["fiscal_period"]}]
        dest = nxt(state, "review_summary")
        return Command(
            update={"budgets": new_budgets, "current_stage": dest, "return_to": None,
                    "ui_results": {}},
            goto=dest)

    return {"counsel": counsel, "allocation": allocation, "budget": budget}
