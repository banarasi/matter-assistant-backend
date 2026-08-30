import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from .config import settings
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


def _sse(stream):
    async def gen():
        async for chunk in stream:
            event = chunk[1] if isinstance(chunk, tuple) else chunk
            yield f"data: {json.dumps(event)}\n\n"
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

    async def gen():
        yield f'data: {json.dumps({"type": "conversation", "conversation_id": cid})}\n\n'
        async for chunk in graph.astream(initial, cfg, stream_mode="custom"):
            event = chunk[1] if isinstance(chunk, tuple) else chunk
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


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
