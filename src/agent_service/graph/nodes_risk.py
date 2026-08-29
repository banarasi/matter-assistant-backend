from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from .. import events
from ..mcp_client import MCPCaller
from ..state import MatterDraft
from .nodes_intake import fetch_values, idem_key, nxt

PIC_FIELDS = ("pic_employee_id", "pic_employee_name")
RISK_FIELDS = ("incident_risk_category", "nfr_taxonomy", "risk_theme")
ALL_FIELDS = PIC_FIELDS + RISK_FIELDS


async def _call(mcp: MCPCaller, writer, tool: str, args: dict) -> dict | None:
    # Same guard shape as nodes_intake.fetch_values: a transport failure (MCP
    # server down/unreachable) is an agent-side concern, not a business error
    # returned by a tool, so it gets its own MCP_UNAVAILABLE event and the
    # caller treats it as a failed call (None) rather than crashing the node.
    try:
        return await mcp.call(tool, args)
    except Exception as e:
        writer(events.error("MCP_UNAVAILABLE", f"{tool} unavailable: {e}", None))
        return None


def make_risk_nodes(model, mcp: MCPCaller):

    async def pic_risk(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("pic_risk"))
        values = {f: getattr(state, f) for f in ALL_FIELDS}
        employees = state.ui_results.get("employees", [])
        c = events.card(
            "PicRiskCard",
            employees=employees,
            values=values,
            incident_risk_categories=await fetch_values(mcp, "incident_risk_category"),
            nfr_taxonomies=await fetch_values(mcp, "nfr_taxonomy"),
            risk_themes=await fetch_values(mcp, "risk_theme"),
        )
        writer(c)
        payload = interrupt(c)
        ptype = payload.get("type")

        if ptype == "action" and payload.get("name") == "search_employee":
            res = await _call(mcp, writer, "search_employees",
                              {"query": payload.get("query", "")})
            if res is None:
                return Command(goto="pic_risk")
            return Command(update={"ui_results": {"employees": res.get("employees", [])}},
                           goto="pic_risk")

        if ptype == "action" and payload.get("name") == "pic_self":
            first = state.requested_by.split(".")[0]
            res = await _call(mcp, writer, "search_employees", {"query": first})
            if res is None:
                return Command(goto="pic_risk")
            found = res.get("employees", [])
            if found:
                emp = found[0]
                return Command(update={"pic_employee_id": emp["id"],
                                       "pic_employee_name": emp["name"]},
                               goto="pic_risk")
            return Command(update={"ui_results": {"employees": found}}, goto="pic_risk")

        if ptype == "card_submit":
            updates = {k: v for k, v in payload.get("values", {}).items() if k in ALL_FIELDS}
            probe = state.model_copy(update=updates)
            missing = [f for f in ALL_FIELDS if not getattr(probe, f)]
            if missing:
                writer(events.error("REQUIRED_FIELD_MISSING",
                                    f"{missing[0]} is required", missing[0]))
                return Command(update=updates, goto="pic_risk")
            res = await _call(mcp, writer, "verify_matter_access",
                              {"employee_id": probe.pic_employee_id})
            if res is None:
                return Command(update=updates, goto="pic_risk")
            if not res.get("ok"):
                e = res["error"]
                writer(events.error(e["code"], e["message"], e.get("field")))
                return Command(update=updates, goto="pic_risk")
            if not res["verified"]:
                writer(events.error(
                    "PIC_NO_ENTITLEMENT",
                    f"{probe.pic_employee_name} has no Passport entitlement — "
                    "please choose another PIC", "pic_employee_id"))
                return Command(update=updates, goto="pic_risk")
            writer(events.card("AccessVerifiedBadge", employee_name=probe.pic_employee_name))
            dest = nxt(state, "create_shell")
            return Command(update={**updates, "access_verified": True,
                                   "current_stage": dest, "return_to": None,
                                   "ui_results": {}},
                           goto=dest)

        # unknown payload type -> re-render
        return Command(goto="pic_risk")

    async def create_shell(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("create_shell"))
        if state.matter_id:  # editing loop revisit: shell already exists
            return Command(goto=nxt(state, "counsel"), update={"return_to": None})
        payload = state.core_payload()
        # NOTE on pattern deviation: create_matter runs BEFORE the node's single
        # interrupt() below, so it WILL re-execute if this node resumes after the
        # MatterCreatedCard interrupt (LangGraph replays the whole node body).
        # That's exactly why idem_key exists: same conversation+node+payload ->
        # the MCP server's idempotency store replays the cached response instead
        # of inserting a second matter row (see _write in passport_mcp/tools.py).
        # We deliberately do NOT call ask() here (unlike other nodes) because its
        # streamed text would also replay as a visible duplicate sentence on
        # resume, and there is no clean way to detect "this is a resume" from
        # inside the node. Instead the confirmation text is a static prop on the
        # card itself (message=...), which is idempotent to re-emit.
        res = await _call(mcp, writer, "create_matter", {
            "payload": payload,
            "idempotency_key": idem_key(state, "create_shell", payload),
            "requested_by": state.requested_by,
            "correlation_id": state.correlation_id,
        })
        if res is None:
            return Command(goto="create_shell")
        if not res.get("ok"):
            e = res["error"]
            writer(events.error(e["code"], e["message"], e.get("field")))
            return Command(goto="basics", update={"current_stage": "basics"})
        matter_id = res["matter_id"]
        c = events.card(
            "MatterCreatedCard",
            matter_id=matter_id,
            message=f"Matter {matter_id} created successfully.",
            actions=["add_org", "configure_budget", "review"],
        )
        writer(c)
        payload_in = interrupt(c)
        name = payload_in.get("name", "continue")
        dest = {"configure_budget": "allocation",
                "review": "review_summary"}.get(name, "counsel")
        return Command(update={"matter_id": matter_id, "current_stage": dest}, goto=dest)

    return {"pic_risk": pic_risk, "create_shell": create_shell}
