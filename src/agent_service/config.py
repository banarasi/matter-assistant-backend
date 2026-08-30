import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    extractor_model: str = os.environ.get("EXTRACTOR_MODEL", "claude-haiku-4-5")
    converse_model: str = os.environ.get("CONVERSE_MODEL", "claude-sonnet-5")
    mcp_server_url: str = os.environ.get("MCP_SERVER_URL", "http://localhost:8081/mcp")
    agent_database_url: str = os.environ.get(
        "AGENT_DATABASE_URL", "postgresql://passport:passport@localhost:5433/agent")
    use_stub_model: bool = os.environ.get("USE_STUB_MODEL", "").lower() in ("1", "true")
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in
        os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()))


settings = Settings()
