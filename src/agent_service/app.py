import asyncio
import json
import logging
import sys
import uuid
import warnings
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

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

from fastapi import FastAPI, Header, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from .config import settings
from .events import error
from .graph.builder import build_graph
from .mcp_client import MCPClient
from .model_client import make_model_client
from .state import MatterDraft

logger = logging.getLogger(__name__)


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextMessage(_Message):
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=10_000)


class CardSubmitMessage(_Message):
    type: Literal["card_submit"]
    values: dict[str, Any] = Field(max_length=100)


class ActionMessage(_Message):
    type: Literal["action"]
    name: str = Field(min_length=1, max_length=100)
    query: str | None = Field(default=None, max_length=500)
    values: dict[str, Any] | None = Field(default=None, max_length=100)


Message = Annotated[
    TextMessage | CardSubmitMessage | ActionMessage,
    Field(discriminator="type"),
]
ConversationId = Annotated[
    str,
    Path(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(settings.agent_database_url) as saver:
        await saver.setup()
        app.state.graph = build_graph(
            make_model_client(settings), MCPClient(settings.mcp_server_url), saver)
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=list(settings.cors_origins),
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def _graph(app: FastAPI):
    return getattr(app.state, "graph_override", None) or app.state.graph


def _sse(stream, first_event: dict | None = None,
         on_close: Callable[[], None] | None = None):
    async def gen():
        try:
            if first_event is not None:
                yield f"data: {json.dumps(first_event)}\n\n"
            try:
                async for chunk in stream:
                    event = chunk[1] if isinstance(chunk, tuple) else chunk
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception:  # the client must ALWAYS receive done
                logger.exception("Conversation stream failed")
                yield f"data: {json.dumps(error('INTERNAL', 'The assistant could not complete this request. Please try again.'))}\n\n"
            yield 'data: {"type": "done"}\n\n'
        finally:
            if on_close is not None:
                on_close()
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-store",
                                      "X-Content-Type-Options": "nosniff"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/conversations")
async def create_conversation(request: Request,
                              x_user_id: str = Header(default="jane.smith")):
    x_user_id = x_user_id.strip()
    if not x_user_id or len(x_user_id) > 200:
        raise HTTPException(status_code=400, detail="X-User-Id must be 1-200 characters")
    cid = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": cid}}
    initial = MatterDraft(conversation_id=cid, correlation_id=cid, requested_by=x_user_id)
    graph = _graph(request.app)
    return _sse(graph.astream(initial, cfg, stream_mode="custom"),
                first_event={"type": "conversation", "conversation_id": cid})


@app.post("/conversations/{cid}/messages")
async def send_message(cid: ConversationId, payload: Message, request: Request):
    cfg = {"configurable": {"thread_id": cid}}
    graph = _graph(request.app)
    snapshot = await graph.aget_state(cfg)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="conversation not found")
    if _pending_card(snapshot) is None:
        raise HTTPException(status_code=409, detail="conversation is not awaiting input")
    active = getattr(request.app.state, "active_conversations", None)
    if active is None:
        active = request.app.state.active_conversations = set()
    if cid in active:
        raise HTTPException(status_code=409, detail="conversation already has a request in progress")
    active.add(cid)
    try:
        stream = graph.astream(
            Command(resume=payload.model_dump(exclude_none=True)), cfg, stream_mode="custom")
        return _sse(stream, on_close=lambda: active.discard(cid))
    except Exception:
        active.discard(cid)
        raise


def _pending_card(snapshot):
    pending = None
    for task in getattr(snapshot, "tasks", []) or []:
        for intr in getattr(task, "interrupts", []) or []:
            pending = intr.value
    return pending


@app.get("/conversations/{cid}/state")
async def get_state(cid: ConversationId, request: Request):
    cfg = {"configurable": {"thread_id": cid}}
    graph = _graph(request.app)
    snapshot = await graph.aget_state(cfg)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="conversation not found")
    pending = _pending_card(snapshot)
    values = snapshot.values
    if hasattr(values, "model_dump"):
        values = values.model_dump()
    return JSONResponse(
        {"values": values, "pending_card": pending},
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_service.app:app", host="0.0.0.0", port=8080, loop=loop_factory)
