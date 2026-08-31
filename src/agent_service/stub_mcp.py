"""In-process stand-in for the Passport MCP server (USE_STUB_MCP=true).

Runs the REAL tool implementations from ``passport_mcp`` (rules engine, error codes,
mock reference data) over an in-memory store, so the whole wizard — picklists,
searches, validation failures, idempotent writes, submission — behaves exactly like
the live server while needing no MCP process and no Postgres. State is process-local
and lost on restart; audit rows are kept only in memory. Dev/demo/UI-testing only.

Requires the sibling package: ``pip install -e ../passport-mcp-server``.
"""
import copy
import threading
from contextlib import contextmanager

try:
    from passport_mcp import refdata, tools
except ImportError as exc:  # pragma: no cover - exercised only in a broken install
    raise ImportError(
        "USE_STUB_MCP needs the passport_mcp package: "
        "pip install -e ../passport-mcp-server") from exc

MATTER_SEQ_START = 1245  # mirrors matter_seq in passport_mcp migrations
TOOLS = frozenset((
    "get_reference_data", "get_required_fields", "search_employees",
    "verify_matter_access", "search_organizations", "search_outside_counsel",
    "search_cost_centers", "get_matter", "create_matter", "add_matter_party",
    "set_cost_allocation", "create_budget", "submit_matter",
))


class InMemoryStore:
    """Drop-in for passport_mcp.stores.Store with dicts instead of tables.

    Uniqueness mirrors the schema: one party per (matter, org, role), one allocation
    row per (matter, cost center), one budget per (matter, org, fiscal period).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._seq = MATTER_SEQ_START
        self.matters: dict[str, dict] = {}
        self.parties: dict[str, list[dict]] = {}
        self.allocations: dict[str, list[dict]] = {}
        self.budgets: dict[str, list[dict]] = {}
        self.audit_log: list[dict] = []
        self.idempotency: dict[str, dict] = {}

    @contextmanager
    def idempotency_lock(self, key: str):
        with self._lock:
            yield

    def create_matter(self, core: dict, created_by: str) -> str:
        with self._lock:
            matter_id = f"MAT-2026-{self._seq:06d}"
            self._seq += 1
            self.matters[matter_id] = {
                "status": "draft", "core": copy.deepcopy(core), "created_by": created_by}
        return matter_id

    def matter_status(self, matter_id: str) -> str | None:
        matter = self.matters.get(matter_id)
        return matter["status"] if matter else None

    def get_matter(self, matter_id: str) -> dict | None:
        matter = self.matters.get(matter_id)
        if matter is None:
            return None
        return {
            "matter_id": matter_id,
            "status": matter["status"],
            "core": copy.deepcopy(matter["core"]),
            "parties": copy.deepcopy(self.parties.get(matter_id, [])),
            "allocations": sorted(copy.deepcopy(self.allocations.get(matter_id, [])),
                                  key=lambda a: a["cc_id"]),
            "budgets": [{**b, "amount": float(b["amount"])}
                        for b in self.budgets.get(matter_id, [])],
        }

    def add_party(self, matter_id: str, org_id: str, org_name: str, role: str) -> None:
        if not self.has_party(matter_id, org_id, role):
            self.parties.setdefault(matter_id, []).append(
                {"org_id": org_id, "org_name": org_name, "role": role})

    def replace_allocations(self, matter_id: str, allocs: list[dict]) -> None:
        self.allocations[matter_id] = [
            {"cc_id": a["cc_id"], "cc_name": a["cc_name"], "pct": a["pct"]} for a in allocs]

    def add_budget(self, matter_id, org_id, amount, currency, fiscal_period):
        if self.budget_exists(matter_id, org_id, fiscal_period):
            return None
        rows = self.budgets.setdefault(matter_id, [])
        rows.append({"org_id": org_id, "amount": amount, "currency": currency,
                     "fiscal_period": fiscal_period})
        return sum(len(r) for r in self.budgets.values())

    def has_party(self, matter_id: str, org_id: str, role: str) -> bool:
        return any(p["org_id"] == org_id and p["role"] == role
                   for p in self.parties.get(matter_id, []))

    def budget_exists(self, matter_id: str, org_id: str, fiscal_period: str) -> bool:
        return any(b["org_id"] == org_id and b["fiscal_period"] == fiscal_period
                   for b in self.budgets.get(matter_id, []))

    def submit(self, matter_id: str) -> bool:
        matter = self.matters.get(matter_id)
        if matter is None or matter["status"] != "draft":
            return False
        matter["status"] = "submitted"
        return True

    def audit(self, requested_by, correlation_id, tool, inputs, outcome, error_code) -> None:
        self.audit_log.append({
            "requested_by": requested_by, "correlation_id": correlation_id, "tool": tool,
            "inputs": copy.deepcopy(inputs), "outcome": outcome, "error_code": error_code})

    def idem_get(self, key: str) -> dict | None:
        cached = self.idempotency.get(key)
        return copy.deepcopy(cached) if cached is not None else None

    def idem_put(self, key: str, response: dict) -> None:
        self.idempotency.setdefault(key, copy.deepcopy(response))


class StubMCPClient:
    """MCPCaller that dispatches straight to passport_mcp.tools over an InMemoryStore."""

    def __init__(self, store: InMemoryStore | None = None):
        self.store = store or InMemoryStore()
        self.ctx = tools.ToolContext(ref=refdata.load(), store=self.store)

    async def call(self, tool: str, args: dict) -> dict:
        if tool not in TOOLS:
            raise RuntimeError(f"unknown MCP tool {tool}")
        # Deep-copy so callers can't mutate what the store retained (JSON semantics).
        return copy.deepcopy(getattr(tools, tool)(self.ctx, **args))
