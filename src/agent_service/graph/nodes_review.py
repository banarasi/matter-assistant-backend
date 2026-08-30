import json

from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.types import Command, interrupt

from .. import events
from ..state import MatterDraft
from .nodes_intake import ask, idem_key

EDIT_TARGETS = {"basics", "jurisdiction", "cond_fields", "pic_risk",
                "counsel", "allocation", "budget"}


async def _get_matter(mcp, writer, matter_id: str) -> dict | None:
    # Same guard shape as nodes_setup._call / nodes_risk._call: a transport
    # failure (MCP server down/unreachable) is an agent-side concern, not a
    # business error returned by a tool, so it gets its own MCP_UNAVAILABLE
    # event and the caller treats it as a failed call (None) rather than
    # crashing the node.
    try:
        snap = await mcp.call("get_matter", {"matter_id": matter_id})
    except Exception as e:
        writer(events.error("MCP_UNAVAILABLE", f"get_matter unavailable: {e}", None))
        return None
    if not snap.get("ok"):
        e = snap["error"]
        writer(events.error(e["code"], e["message"], e.get("field")))
        return None
    return snap


def make_review_nodes(model, mcp):

    async def review_summary(state: MatterDraft) -> Command:
        # No interrupt() here -> this node's body never replays. It runs
        # exactly once on first arrival at review and once per completed
        # edit loop (return_to routes edits back to review_summary, not
        # review), so the Sonnet-class narrative streams exactly once per
        # visit instead of re-streaming on every ReviewCard resume.
        writer = get_stream_writer()
        writer(events.stage("review"))
        snap = await _get_matter(mcp, writer, state.matter_id)
        if snap is None:
            return Command(goto="basics", update={"current_stage": "basics"})
        await ask(model, writer,
                  "You are the Enterprise Matter Assistant. Summarize the matter "
                  "configuration for final review in 2-3 sentences.",
                  json.dumps(snap))
        return Command(update={"current_stage": "review"}, goto="review")

    async def review(state: MatterDraft) -> Command:
        # Exactly ONE interrupt() per execution: on resume, only the snapshot
        # fetch + card emit below replay (a benign duplicate card), never the
        # narrative (that lives in review_summary, which has no interrupt).
        writer = get_stream_writer()
        snap = await _get_matter(mcp, writer, state.matter_id)
        if snap is None:
            return Command(goto="basics", update={"current_stage": "basics"})
        c = events.card("ReviewCard", snapshot=snap)
        writer(c)
        payload = interrupt(c)
        ptype = payload.get("type")
        name = payload.get("name", "")
        if ptype == "action" and name.startswith("edit:"):
            target = name.split(":", 1)[1]
            if target in EDIT_TARGETS:
                return Command(
                    update={"return_to": "review_summary", "current_stage": target,
                            "write_seq": state.write_seq + 1},
                    goto=target)
        if ptype == "action" and name == "confirm_submit":
            return Command(update={"current_stage": "submit"}, goto="submit")
        # anything else -> re-render card only, no re-summary
        return Command(goto="review")

    async def submit(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("submit"))
        # Same transport guard as _get_matter/_call: an MCP outage during
        # submission must land the user back on review, not crash the node.
        try:
            res = await mcp.call("submit_matter", {
                "matter_id": state.matter_id,
                "idempotency_key": idem_key(state, "submit"),
                "requested_by": state.requested_by,
                "correlation_id": state.correlation_id})
        except Exception as e:
            writer(events.error("MCP_UNAVAILABLE", f"submission failed: {e}", None))
            return Command(goto="review", update={"current_stage": "review"})
        if not res.get("ok"):
            e = res["error"]
            writer(events.error(e["code"], e["message"], e.get("field")))
            return Command(goto="review", update={"current_stage": "review"})
        writer(events.card("SubmittedCard", matter_id=state.matter_id,
                           message="You will receive a confirmation email shortly."))
        writer(events.stage("submitted"))
        return Command(update={"submitted": True, "current_stage": "submitted"}, goto=END)

    return {"review_summary": review_summary, "review": review, "submit": submit}
