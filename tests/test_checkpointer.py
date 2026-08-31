"""AGENT_CHECKPOINTER selects where LangGraph thread state lives.

``postgres`` (default) needs AGENT_DATABASE_URL; ``memory`` keeps state in-process so
the service can be brought up without Docker/Postgres (demo/dev only).
"""
import os
from dataclasses import replace

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

import agent_service.app as app_module
from agent_service.app import app, lifespan
from agent_service.config import Settings

from .test_app import parse_sse


def test_checkpointer_defaults_to_postgres(monkeypatch):
    monkeypatch.delenv("AGENT_CHECKPOINTER", raising=False)
    assert Settings().checkpointer == "postgres"


@pytest.mark.parametrize("raw", ["memory", "MEMORY", " memory "])
def test_checkpointer_memory_from_env(monkeypatch, raw):
    monkeypatch.setenv("AGENT_CHECKPOINTER", raw)
    assert Settings().checkpointer == "memory"


def test_checkpointer_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("AGENT_CHECKPOINTER", "sqlite")
    with pytest.raises(ValueError, match="AGENT_CHECKPOINTER"):
        Settings()


@pytest.fixture()
async def memory_app(monkeypatch):
    monkeypatch.setattr(
        app_module, "settings",
        replace(app_module.settings, checkpointer="memory", use_stub_model=True))
    async with lifespan(app):
        yield app


async def test_lifespan_memory_mode_builds_graph_without_postgres(memory_app):
    assert isinstance(memory_app.state.graph.checkpointer, MemorySaver)


async def test_memory_mode_serves_a_conversation(memory_app):
    transport = httpx.ASGITransport(app=memory_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (await c.get("/healthz")).json() == {"ok": True}
        r = await c.post("/conversations")
        events = parse_sse(r.text)
        cid = events[0]["conversation_id"]
        assert any(e.get("card") == "WelcomeCard" for e in events)
        assert events[-1]["type"] == "done"
        r = await c.get(f"/conversations/{cid}/state")
        assert r.status_code == 200
        assert r.json()["pending_card"]["card"] == "WelcomeCard"


def test_load_dotenv_sets_only_missing_keys(tmp_path, monkeypatch):
    from agent_service.config import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nAGENT_T_NEW=from-file\nAGENT_T_KEPT=from-file\n"
        "AGENT_T_QUOTED='a=b'\nnot-a-pair\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_T_NEW", raising=False)
    monkeypatch.setenv("AGENT_T_KEPT", "from-env")
    monkeypatch.delenv("AGENT_T_QUOTED", raising=False)
    load_dotenv(env_file)
    assert os.environ["AGENT_T_NEW"] == "from-file"
    assert os.environ["AGENT_T_KEPT"] == "from-env"
    assert os.environ["AGENT_T_QUOTED"] == "a=b"
    for k in ("AGENT_T_NEW", "AGENT_T_QUOTED"):
        monkeypatch.delenv(k)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    from agent_service.config import load_dotenv

    load_dotenv(tmp_path / "absent.env")  # must not raise
