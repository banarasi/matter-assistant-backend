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

    def core_payload(self) -> dict:
        keys = ["matter_name", "pabu", "matter_type", "matter_subtype", "country",
                "state_region", "business_segment", "legal_entity",
                "confidentiality_class", "pic_employee_id", "incident_risk_category",
                "nfr_taxonomy", "risk_theme"]
        return {k: getattr(self, k) for k in keys}
