from agent_service.state import MatterDraft
from agent_service import validation


def test_defaults():
    s = MatterDraft()
    assert s.current_stage == "welcome" and s.requested_by == "jane.smith"
    assert s.allocations == [] and not s.submitted


def test_core_payload_shape():
    s = MatterDraft(matter_name="X", pabu="PABU-EMP", matter_type="MT-EMP-INV",
                    matter_subtype="MST-POL", country="CH", business_segment="BS-WM",
                    legal_entity="LE-UBS-CH", confidentiality_class="CC-CONF",
                    pic_employee_id="E1001", incident_risk_category="IRC-HIGH",
                    nfr_taxonomy="NFR-EP", risk_theme="RT-PG")
    p = s.core_payload()
    assert p["matter_name"] == "X" and p["pic_employee_id"] == "E1001"
    assert set(p) == {
        "matter_name", "pabu", "matter_type", "matter_subtype", "country",
        "state_region", "business_segment", "legal_entity", "confidentiality_class",
        "pic_employee_id", "incident_risk_category", "nfr_taxonomy", "risk_theme",
    }


def test_missing_basics():
    s = MatterDraft(matter_name="X", pabu="PABU-EMP")
    assert validation.missing_basics(s) == ["matter_type", "matter_subtype"]


def test_missing_cond_fields():
    s = MatterDraft(required_extra_fields=["business_segment", "legal_entity"],
                    business_segment="BS-WM")
    assert validation.missing_cond_fields(s) == ["legal_entity"]


def test_allocations_total():
    assert validation.allocations_total([{"pct": 70}, {"pct": 30}]) == 100
