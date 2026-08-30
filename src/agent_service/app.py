import asyncio
import json
import sys
import uuid
import warnings
from contextlib import asynccontextmanager

if sys.platform == "win32":
    # psycopg async requires a selector event loop on Windows (Proactor is unsupported).
    # Policies are deprecated on Python 3.14 (removal slated for 3.16); silence just
    # that warning here. NOTE: uvicorn >= 0.36 ignores policies and passes an explicit
    # loop_factory (Proactor on win32), so on Windows run the service via
    # `python -m agent_service.app` or `uvicorn ... --loop agent_service.app:loop_factory`
    # (see loop_factory below); the policy still covers plain asyncio.run() embedders.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def loop_factory() -> asyncio.AbstractEventLoop:
    """Zero-arg loop factory for uvicorn's --loop: selector loop so psycopg async works on Windows."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from .config import settings
from .events import error
from .graph.builder import build_graph
from .mcp_client import MCPClient
from .model_client import make_model_client
from .state import MatterDraft


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(settings.agent_database_url) as saver:
        await saver.setup()
        app.state.graph = build_graph(
            make_model_client(settings), MCPClient(settings.mcp_server_url), saver)
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


def _graph(app: FastAPI):
    return getattr(app.state, "graph_override", None) or app.state.graph


def _sse(stream, first_event: dict | None = None):
    async def gen():
        if first_event is not None:
            yield f"data: {json.dumps(first_event)}\n\n"
        try:
            async for chunk in stream:
                event = chunk[1] if isinstance(chunk, tuple) else chunk
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # the client must ALWAYS receive done
            yield f"data: {json.dumps(error('INTERNAL', str(e)))}\n\n"
        yield 'data: {"type": "done"}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/conversations")
async def create_conversation(request: Request,
                              x_user_id: str = Header(default="jane.smith")):
    cid = uuid.uuid4().hex[:12]
    cfg = {"configurable": {"thread_id": cid}}
    initial = MatterDraft(conversation_id=cid, correlation_id=cid, requested_by=x_user_id)
    graph = _graph(request.app)
    return _sse(graph.astream(initial, cfg, stream_mode="custom"),
                first_event={"type": "conversation", "conversation_id": cid})


@app.post("/conversations/{cid}/messages")
async def send_message(cid: str, request: Request):
    payload = await request.json()
    cfg = {"configurable": {"thread_id": cid}}
    graph = _graph(request.app)
    return _sse(graph.astream(Command(resume=payload), cfg, stream_mode="custom"))


@app.get("/conversations/{cid}/state")
async def get_state(cid: str, request: Request):
    cfg = {"configurable": {"thread_id": cid}}
    graph = _graph(request.app)
    snapshot = await graph.aget_state(cfg)
    pending = None
    for task in getattr(snapshot, "tasks", []) or []:
        for intr in getattr(task, "interrupts", []) or []:
            pending = intr.value
    values = snapshot.values
    if hasattr(values, "model_dump"):
        values = values.model_dump()
    return {"values": values, "pending_card": pending}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_service.app:app", host="0.0.0.0", port=8080, loop=loop_factory)
