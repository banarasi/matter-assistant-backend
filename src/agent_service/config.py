import os
from dataclasses import dataclass, field
from pathlib import Path

CHECKPOINTERS = ("postgres", "memory")


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Populate os.environ from a KEY=VALUE .env file without overriding real env vars.

    Lets `python -m agent_service.app` (no Docker) pick up the same .env that
    `docker compose --env-file .env` uses. Path: AGENT_ENV_FILE, else ./.env.
    """
    path = Path(path or os.environ.get("AGENT_ENV_FILE", ".env"))
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


load_dotenv()


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true")


def _checkpointer_from_env() -> str:
    raw = os.environ.get("AGENT_CHECKPOINTER", "postgres").strip().lower()
    if raw not in CHECKPOINTERS:
        raise ValueError(
            f"AGENT_CHECKPOINTER must be one of {', '.join(CHECKPOINTERS)}; got {raw!r}")
    return raw


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    extractor_model: str = os.environ.get("EXTRACTOR_MODEL", "claude-haiku-4-5")
    converse_model: str = os.environ.get("CONVERSE_MODEL", "claude-sonnet-5")
    mcp_server_url: str = os.environ.get("MCP_SERVER_URL", "http://localhost:8081/mcp")
    agent_database_url: str = os.environ.get(
        "AGENT_DATABASE_URL", "postgresql://passport:passport@localhost:5433/agent")
    # "postgres" (default; durable, needs AGENT_DATABASE_URL) or "memory" (in-process,
    # lost on restart — lets the service run with no Docker/Postgres for dev/demo).
    checkpointer: str = field(default_factory=_checkpointer_from_env)
    use_stub_model: bool = os.environ.get("USE_STUB_MODEL", "").lower() in ("1", "true")
    # In-process Passport tools over an in-memory store instead of MCP_SERVER_URL —
    # no MCP server/Postgres needed (needs `pip install -e ../passport-mcp-server`).
    use_stub_mcp: bool = field(default_factory=lambda: _flag("USE_STUB_MCP"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in
        os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()))


settings = Settings()
