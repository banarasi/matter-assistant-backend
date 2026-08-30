# Matter Assistant — Backend (agent service)

The backend of the Enterprise Matter Assistant: a FastAPI SSE BFF driving a
LangGraph wizard that walks a requester through creating a legal matter, calling
the Passport MCP server for every read/write and Claude for extraction and
conversation (`claude-haiku-4-5` / `claude-sonnet-5` via a swappable
`ModelClient`; `USE_STUB_MODEL=true` runs without any API key).

Extracted from the `legalDashboard` monorepo with history preserved. Sibling repos:

| Repo | Role |
|---|---|
| `passport-mcp-server` | Tool contract + business rules + persistence |
| **matter-assistant-backend** (this) | LangGraph wizard agent / SSE BFF |
| `matter-assistant-ui` | Next.js chat portal |

## Run the full demo stack

This repo carries the orchestration. With all three repos checked out as
siblings (as in `C:\ai_apps\Demo`):

```bash
cp .env.example .env        # keep USE_STUB_MODEL=true, or set a real ANTHROPIC_API_KEY
docker compose --env-file .env up -d --build
# portal on :3000, agent API on :8080, MCP on :8081, Postgres on :5433
```

## Develop & test

```bash
docker compose up -d postgres                  # tests only need the database
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ../passport-mcp-server          # tests run the real MCP tools in-process
python -m pytest tests/ -q                     # deterministic: stub model + fake MCP transport
```

Natively on Windows, start the server with `python -m agent_service.app`
(plain `uvicorn` breaks — proactor event loop vs psycopg).

## Invariants

- One `interrupt()` per LangGraph node execution; loops via
  `Command(update, goto=<same node>)` — see the comment atop
  `src/agent_service/graph/nodes_intake.py`.
- The server (MCP) is authoritative for business validation; agent-side checks
  are UX sugar. The model never selects tools or routes.
- SSE event shapes in `src/agent_service/events.py` mirror the portal's
  `src/lib/types.ts`; every turn stream ends with `{"type":"done"}`.
- Never pass `temperature`/`top_p`/`top_k` to the models.
