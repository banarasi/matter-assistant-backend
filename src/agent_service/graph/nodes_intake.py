import hashlib
import json
import logging

from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from .. import events, validation
from ..mcp_client import MCPCaller
from ..model_client import ModelClient
from ..state import MatterDraft

logger = logging.getLogger(__name__)

STEPS = ["Matter Details", "Risk & Classification", "Organizations & Counsel",
         "Budgets & Cost Centers", "Review & Submit"]

EXTRACT_SYSTEM = (
    "Extract matter-intake fields from the user's message. Only fill fields the "
    "user actually stated; leave everything else null. Labels must be copied "
    "verbatim where possible."
)

BASICS_FIELDS = ("matter_name", "pabu", "matter_type", "matter_subtype")
COND_FIELDS = ("business_segment", "legal_entity", "confidentiality_class")

# Node pattern: exactly ONE interrupt() per node execution. LangGraph replays a
# node body from the top on every resume (already-resolved interrupts return
# cached values, but writer()/MCP/LLM calls re-run), so looping inside the node
# across multiple interrupts would replay every prior iteration's events and
# side effects. Instead, each node renders its card, interrupts once, processes
# the payload, and loops VIA THE GRAPH: invalid or partial input returns
# Command(update=<merged fields>, goto=<same node>) so the re-executed node
# re-renders the card from updated state. Per resume this costs at most one
# benign duplicate of the node's own card; errors, text deltas, and LLM calls
# run exactly once per turn.


def idem_key(state: MatterDraft, node: str, payload: dict | None = None) -> str:
    raw = json.dumps({"c": state.conversation_id, "n": node, "p": payload or {},
                      "s": getattr(state, "write_seq", 0)}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def nxt(state: MatterDraft, default: str) -> str:
    return state.return_to or default


async def ask(model: ModelClient, writer, system: str, user: str) -> None:
    try:
        async for delta in model.converse_stream(system, user):
            writer(events.text_delta(delta))
    except Exception:
        report_model_failure(writer, "conversation")


def report_model_failure(writer, operation: str) -> None:
    logger.exception("Model %s failed", operation)
    writer(events.error(
        "MODEL_UNAVAILABLE",
        "The language model is temporarily unavailable; your saved form data is intact.",
    ))


def report_mcp_failure(writer, operation: str) -> None:
    logger.exception("MCP operation %s failed", operation)
    writer(events.error(
        "MCP_UNAVAILABLE",
        "The data service is temporarily unavailable. Please try again.",
        None,
    ))


def who(state: MatterDraft) -> dict:
    return {"requested_by": state.requested_by, "correlation_id": state.correlation_id}


async def fetch_values(mcp: MCPCaller, domain: str, parent: str | None = None,
                       state: MatterDraft | None = None) -> list:
    # MCP_UNAVAILABLE is an agent-side transport error code: it is emitted here
    # (not by the MCP server) when the reference-data call itself fails, so the
    # turn degrades to an empty picklist instead of crashing.
    try:
        args = {"domain": domain, "parent": parent}
        if state is not None:
            args.update(who(state))
        res = await mcp.call("get_reference_data", args)
    except Exception:
        writer = get_stream_writer()
        report_mcp_failure(writer, "get_reference_data")
        return []
    return res.get("values", []) if res.get("ok") else []


def match_label(values: list[dict], label: str | None) -> str | None:
    if not label:
        return None
    low = label.strip().lower()
    if not low:
        return None
    exact = next((v for v in values if v["label"].lower() == low), None)
    if exact:
        return exact["id"]
    partial = [v for v in values if low in v["label"].lower()]
    return partial[0]["id"] if len(partial) == 1 else None


class BasicsExtract(BaseModel):
    matter_name: str | None = None
    pabu_label: str | None = None
    matter_type_label: str | None = None
    matter_subtype_label: str | None = None


def make_intake_nodes(model: ModelClient, mcp: MCPCaller):

    async def welcome(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("welcome"))
        c = events.card("WelcomeCard", steps=STEPS)
        writer(c)
        interrupt(c)
        return Command(update={"current_stage": "basics"}, goto="basics")

    async def basics(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("basics"))
        values = {f: getattr(state, f) for f in BASICS_FIELDS}
        pabu = await fetch_values(mcp, "pabu", state=state)
        mtypes = await fetch_values(mcp, "matter_type", values["pabu"], state)
        msubs = await fetch_values(mcp, "matter_subtype", values["matter_type"], state)
        missing = [f for f in BASICS_FIELDS if not values[f]]
        c = events.card("BasicInfoCard", values=values, pabu=pabu,
                        matter_types=mtypes, matter_subtypes=msubs, missing=missing)
        writer(c)
        payload = interrupt(c)
        if payload.get("type") == "text":
            updates: dict = {}
            try:
                extracted = await model.extract(BasicsExtract, EXTRACT_SYSTEM,
                                                payload["text"])
            except Exception:
                report_model_failure(writer, "extraction")
                return Command(goto="basics")
            if extracted:
                if extracted.matter_name:
                    updates["matter_name"] = extracted.matter_name
                all_types = await fetch_values(mcp, "matter_type", state=state)
                all_subs = await fetch_values(mcp, "matter_subtype", state=state)
                for field, matched in (
                    ("pabu", match_label(pabu, extracted.pabu_label)),
                    ("matter_type", match_label(all_types, extracted.matter_type_label)),
                    ("matter_subtype", match_label(all_subs, extracted.matter_subtype_label)),
                ):
                    if matched:
                        updates[field] = matched
            still = [f for f in BASICS_FIELDS if not (updates.get(f) or values[f])]
            if still:
                await ask(model, writer,
                          "You are the Enterprise Matter Assistant. One short sentence.",
                          f"Ask the user to provide: {', '.join(still)}")
            return Command(update=updates, goto="basics")
        if payload.get("type") == "card_submit":
            updates = {k: v for k, v in payload["values"].items() if k in BASICS_FIELDS}
            probe = state.model_copy(update=updates)
            missing = validation.missing_basics(probe)
            if missing:
                writer(events.error("REQUIRED_FIELD_MISSING",
                                    f"{missing[0]} is required", missing[0]))
                return Command(update=updates, goto="basics")
            dest = nxt(state, "jurisdiction")
            return Command(update={**updates, "current_stage": dest, "return_to": None},
                           goto=dest)
        # unknown payload type -> re-render
        return Command(goto="basics")

    async def jurisdiction(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("jurisdiction"))
        countries = await fetch_values(mcp, "country", state=state)
        c = events.card("JurisdictionCard", countries=countries,
                        values={"country": state.country,
                                "state_region": state.state_region})
        writer(c)
        payload = interrupt(c)
        if payload.get("type") != "card_submit" or not payload["values"].get("country"):
            writer(events.error("REQUIRED_FIELD_MISSING", "country is required", "country"))
            return Command(goto="jurisdiction")
        country = payload["values"]["country"]
        region = payload["values"].get("state_region")
        try:
            res = await mcp.call("get_required_fields", {"country": country, **who(state)})
        except Exception:
            report_mcp_failure(writer, "get_required_fields")
            return Command(update={"country": country, "state_region": region},
                           goto="jurisdiction")
        if not res.get("ok"):
            err = res["error"]
            writer(events.error(err["code"], err["message"], err.get("field")))
            return Command(update={"country": country, "state_region": region},
                           goto="jurisdiction")
        fields = res.get("fields", [])
        if fields:
            writer(events.card("AdditionalInfoNotice", fields=fields))
        dest = nxt(state, "cond_fields" if fields else "pic_risk")
        return Command(update={"country": country, "state_region": region,
                               "required_extra_fields": fields,
                               "current_stage": dest, "return_to": None},
                       goto=dest)

    async def cond_fields(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("cond_fields"))
        values = {f: getattr(state, f) for f in COND_FIELDS}
        c = events.card(
            "AdditionalFieldsCard",
            required=state.required_extra_fields,
            business_segments=await fetch_values(mcp, "business_segment", state=state),
            legal_entities=await fetch_values(mcp, "legal_entity", state=state),
            confidentiality_classes=await fetch_values(
                mcp, "confidentiality_class", state=state),
            values=values,
        )
        writer(c)
        payload = interrupt(c)
        if payload.get("type") == "card_submit":
            updates = {k: v for k, v in payload["values"].items() if k in COND_FIELDS}
            probe = state.model_copy(update=updates)
            missing = validation.missing_cond_fields(probe)
            if missing:
                writer(events.error("REQUIRED_FIELD_MISSING",
                                    f"{missing[0]} is required", missing[0]))
                return Command(update=updates, goto="cond_fields")
            dest = nxt(state, "pic_risk")
            return Command(update={**updates, "current_stage": dest, "return_to": None},
                           goto=dest)
        return Command(goto="cond_fields")

    return {"welcome": welcome, "basics": basics, "jurisdiction": jurisdiction,
            "cond_fields": cond_fields}
