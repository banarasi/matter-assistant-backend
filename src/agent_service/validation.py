"""Local mirrors of MCP rules for instant in-UI feedback. The MCP server is authoritative."""
from .state import MatterDraft

BASICS = ["matter_name", "pabu", "matter_type", "matter_subtype"]


def missing_basics(state: MatterDraft) -> list[str]:
    return [f for f in BASICS if not getattr(state, f)]


def missing_cond_fields(state: MatterDraft) -> list[str]:
    return [f for f in state.required_extra_fields if not getattr(state, f)]


def allocations_total(allocs: list[dict]) -> int:
    return sum(int(a.get("pct") or 0) for a in allocs)
