import pytest
from passport_mcp import db, refdata, tools
from passport_mcp.stores import Store

from .fakes import FakeMCPClient

TABLES = ["audit_log", "idempotency_keys", "budgets", "allocations", "matter_parties", "matters"]


@pytest.fixture(scope="session")
def pool():
    p = db.get_pool()
    db.migrate(p)
    return p


@pytest.fixture()
def fake_mcp(pool):
    with pool.connection() as conn:
        for t in TABLES:
            conn.execute(f"TRUNCATE {t} CASCADE")
        # matter_seq is a free-standing sequence (not owned by any identity column),
        # so TRUNCATE never resets it. Reset it explicitly for deterministic,
        # order-independent matter IDs across test runs (mirrors mcp-server's conftest).
        conn.execute("ALTER SEQUENCE matter_seq RESTART WITH 1245")
    return FakeMCPClient(tools.ToolContext(ref=refdata.load(), store=Store(pool)))
