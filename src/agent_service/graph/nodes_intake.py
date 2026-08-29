import hashlib
import json

from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from .. import events, validation
from ..mcp_client import MCPCaller
from ..model_client import ModelClient
from ..state import MatterDraft

STEPS = ["Matter Details", "Risk & Classification", "Organizations & Counsel",
         "Budgets & Cost Centers", "Review & Submit"]

EXTRACT_SYSTEM = (
    "Extract matter-intake fields from the user's message. Only fill fields the "
    "user actually stated; leave everything else null. Labels must be copied "
    "verbatim where possible."
)


def idem_key(state: MatterDraft, node: str, payload: dict | None = None) -> str:
    raw = json.dumps({"c": state.conversation_id, "n": node, "p": payload or {}},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def nxt(state: MatterDraft, default: str) -> str:
    return state.return_to or default


async def ask(model: ModelClient, writer, system: str, user: str) -> None:
    async for delta in model.converse_stream(system, user):
        writer(events.text_delta(delta))


async def fetch_values(mcp: MCPCaller, domain: str, parent: str | None = None) -> list:
    res = await mcp.call("get_reference_data", {"domain": domain, "parent": parent})
    return res.get("values", []) if res.get("ok") else []


def match_label(values: list[dict], label: str | None) -> str | None:
    if not label:
        return None
    low = label.strip().lower()
    for v in values:
        if v["label"].lower() == low or low in v["label"].lower():
            return v["id"]
    return None


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
        values = {f: getattr(state, f) for f in
                  ("matter_name", "pabu", "matter_type", "matter_subtype")}
        while True:
            pabu = await fetch_values(mcp, "pabu")
            mtypes = await fetch_values(mcp, "matter_type", values["pabu"])
            msubs = await fetch_values(mcp, "matter_subtype", values["matter_type"])
            missing = [f for f in ("matter_name", "pabu", "matter_type", "matter_subtype")
                       if not values[f]]
            c = events.card("BasicInfoCard", values=values, pabu=pabu,
                            matter_types=mtypes, matter_subtypes=msubs, missing=missing)
            writer(c)
            payload = interrupt(c)
            if payload.get("type") == "text":
                extracted = await model.extract(BasicsExtract, EXTRACT_SYSTEM,
                                                payload["text"])
                if extracted:
                    if extracted.matter_name:
                        values["matter_name"] = extracted.matter_name
                    all_types = await fetch_values(mcp, "matter_type")
                    all_subs = await fetch_values(mcp, "matter_subtype")
                    values["pabu"] = match_label(pabu, extracted.pabu_label) or values["pabu"]
                    values["matter_type"] = (match_label(all_types, extracted.matter_type_label)
                                             or values["matter_type"])
                    values["matter_subtype"] = (match_label(all_subs, extracted.matter_subtype_label)
                                                or values["matter_subtype"])
                still = [f for f in ("matter_name", "pabu", "matter_type", "matter_subtype")
                         if not values[f]]
                if still:
                    await ask(model, writer,
                              "You are the Enterprise Matter Assistant. One short sentence.",
                              f"Ask the user to provide: {', '.join(still)}")
                continue
            if payload.get("type") == "card_submit":
                values.update({k: v for k, v in payload["values"].items() if k in values})
                probe = MatterDraft(**{**state.model_dump(), **values})
                missing = validation.missing_basics(probe)
                if missing:
                    writer(events.error("REQUIRED_FIELD_MISSING",
                                        f"{missing[0]} is required", missing[0]))
                    continue
                dest = nxt(state, "jurisdiction")
                return Command(update={**values, "current_stage": dest, "return_to": None},
                               goto=dest)
            # unknown action -> re-render
            continue

    async def jurisdiction(state: MatterDraft) -> Command:
        writer = get_stream_writer()
        writer(events.stage("jurisdiction"))
        while True:
            countries = await fetch_values(mcp, "country")
            c = events.card("JurisdictionCard", countries=countries,
                            values={"country": state.country,
                                    "state_region": state.state_region})
            writer(c)
            payload = interrupt(c)
            if payload.get("type") != "card_submit" or not payload["values"].get("country"):
                writer(events.error("REQUIRED_FIELD_MISSING", "country is required", "country"))
                continue
            country = payload["values"]["country"]
            region = payload["values"].get("state_region")
            res = await mcp.call("get_required_fields", {"country": country})
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
        values = {f: getattr(state, f) for f in
                  ("business_segment", "legal_entity", "confidentiality_class")}
        while True:
            c = events.card(
                "AdditionalFieldsCard",
                required=state.required_extra_fields,
                business_segments=await fetch_values(mcp, "business_segment"),
                legal_entities=await fetch_values(mcp, "legal_entity"),
                confidentiality_classes=await fetch_values(mcp, "confidentiality_class"),
                values=values,
            )
            writer(c)
            payload = interrupt(c)
            if payload.get("type") == "card_submit":
                values.update({k: v for k, v in payload["values"].items() if k in values})
                probe = MatterDraft(**{**state.model_dump(), **values})
                missing = validation.missing_cond_fields(probe)
                if missing:
                    writer(events.error("REQUIRED_FIELD_MISSING",
                                        f"{missing[0]} is required", missing[0]))
                    continue
                dest = nxt(state, "pic_risk")
                return Command(update={**values, "current_stage": dest, "return_to": None},
                               goto=dest)
            continue

    return {"welcome": welcome, "basics": basics, "jurisdiction": jurisdiction,
            "cond_fields": cond_fields}
