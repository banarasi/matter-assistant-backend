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

## Run the backend without Docker

The only hard infrastructure dependency of this service is the LangGraph
checkpointer, which defaults to Postgres. `AGENT_CHECKPOINTER=memory` swaps it
for an in-process `MemorySaver`, so the agent API comes up with nothing but a
Python venv:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                           # then set AGENT_CHECKPOINTER=memory
python -m agent_service.app                    # agent API on :8080
curl http://localhost:8080/healthz             # {"ok":true}
```

`python -m agent_service.app` reads `./.env` on startup (override the path with
`AGENT_ENV_FILE`); variables already set in the shell always win, so a one-off
`AGENT_CHECKPOINTER=memory python -m agent_service.app` works too. Keep
`USE_STUB_MODEL=true` for a key-less run, or set a real `ANTHROPIC_API_KEY`.

What the memory checkpointer changes:

- Conversation state lives in the server process — every restart forgets all
  in-flight conversations (the portal gets `404 conversation not found`).
- Single-process only; do not run more than one worker/replica in this mode.
- No `AGENT_DATABASE_URL`, no `agent` database, no psycopg connection at all
  (the Postgres saver is imported lazily). `docker compose` is unaffected: it
  never forwards `AGENT_CHECKPOINTER`, so containers always use Postgres.

### Without the MCP server too (`USE_STUB_MCP=true`)

By default the wizard still calls the Passport MCP server for every
picklist/read/write (`MCP_SERVER_URL`, default `http://localhost:8081/mcp`), and
that server needs its own Postgres. `USE_STUB_MCP=true` replaces the network
client with `src/agent_service/stub_mcp.py`: the **real** tool implementations
from `passport_mcp` (rules engine, error codes, mock reference data) running
in-process over an in-memory store. Every card, search, validation error,
idempotent write and the final submission behave exactly as against the live
server — the only differences are that matters/audit rows live in the agent
process (lost on restart) and `MCP_SERVER_URL` is ignored.

```bash
pip install -e ../passport-mcp-server   # the stub imports passport_mcp
# .env: AGENT_CHECKPOINTER=memory, USE_STUB_MCP=true, USE_STUB_MODEL=true
python -m agent_service.app             # agent API on :8080, nothing else needed
```

This is the setup for exercising the portal (`../matter-assistant-ui`,
`npm run dev` with `NEXT_PUBLIC_AGENT_URL=http://localhost:8080`) with no
Docker at all. The three stubs are independent: e.g. keep `USE_STUB_MCP=true`
but set a real `ANTHROPIC_API_KEY` (and `USE_STUB_MODEL=`) to test the live
models against canned Passport data.

| Env | Default | Effect |
|---|---|---|
| `AGENT_CHECKPOINTER` | `postgres` | `memory` = LangGraph state in-process, no `AGENT_DATABASE_URL` |
| `USE_STUB_MCP` | (off) | `true` = in-process Passport tools, no MCP server / Postgres |
| `USE_STUB_MODEL` | (off) | `true` = deterministic `StubModelClient`, no Anthropic calls |

Without the stub, run the MCP server natively against a locally installed
Postgres (see `../passport-mcp-server/README.md`, `DATABASE_URL`), or point
`MCP_SERVER_URL` at any reachable instance.

Tests that use the `fake_mcp` fixture still need the Passport Postgres
(`docker compose up -d postgres` or a local install on `:5433`); the
checkpointer/config/app-level tests do not:

```bash
python -m pytest tests/test_checkpointer.py tests/test_stub_mcp.py tests/test_model_client.py tests/test_state.py -q
```

## Invariants

- One `interrupt()` per LangGraph node execution; loops via
  `Command(update, goto=<same node>)` — see the comment atop
  `src/agent_service/graph/nodes_intake.py`.
- The server (MCP) is authoritative for business validation; agent-side checks
  are UX sugar. The model never selects tools or routes.
- SSE event shapes in `src/agent_service/events.py` mirror the portal's
  `src/lib/types.ts`; every turn stream ends with `{"type":"done"}`.
- Never pass `temperature`/`top_p`/`top_k` to the models.
