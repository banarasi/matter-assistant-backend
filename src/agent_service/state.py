from pydantic import BaseModel, Field


class MatterDraft(BaseModel):
    conversation_id: str = ""
    correlation_id: str = ""
    requested_by: str = "jane.smith"
    current_stage: str = "welcome"

    matter_name: str | None = None
    pabu: str | None = None
    matter_type: str | None = None
    matter_subtype: str | None = None
    country: str | None = None
    state_region: str | None = None
    business_segment: str | None = None
    legal_entity: str | None = None
    confidentiality_class: str | None = None
    required_extra_fields: list[str] = Field(default_factory=list)

    pic_self: bool | None = None
    pic_employee_id: str | None = None
    pic_employee_name: str | None = None
    access_verified: bool = False
    incident_risk_category: str | None = None
    nfr_taxonomy: str | None = None
    risk_theme: str | None = None

    matter_id: str | None = None
    organization_id: str | None = None
    organization_name: str | None = None
    outside_counsel_id: str | None = None
    outside_counsel_name: str | None = None
    allocations: list[dict] = Field(default_factory=list)
    budgets: list[dict] = Field(default_factory=list)
    submitted: bool = False
    return_to: str | None = None
    # Transient per-turn scratch space (e.g. employee search results) that must
    # survive a goto-self loop within one node but is never part of the matter
    # payload sent to the MCP server. Cleared ({}) whenever a node hands off to
    # the next stage.
    ui_results: dict = Field(default_factory=dict)
    # Bumped on every review-edit revisit (see nodes_review.py). Folded into
    # idem_key so content-derived idempotency keys stay stable across a node's
    # own LangGraph replays (same key each time the node body re-runs before
    # its interrupt resolves) but change across A->B->A edit cycles, where the
    # user resubmits byte-identical content that was already cached server
    # side from an earlier visit. NOT part of core_payload: it is bookkeeping
    # for the agent's write plumbing, not matter data.
    write_seq: int = 0

    def core_payload(self) -> dict:
        keys = ["matter_name", "pabu", "matter_type", "matter_subtype", "country",
                "state_region", "business_segment", "legal_entity",
                "confidentiality_class", "pic_employee_id", "incident_risk_category",
                "nfr_taxonomy", "risk_theme"]
        return {k: getattr(self, k) for k in keys}
