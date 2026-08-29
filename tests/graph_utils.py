from langgraph.types import Command

from agent_service.state import MatterDraft


async def collect(stream) -> list[dict]:
    out = []
    async for chunk in stream:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            # (mode, payload) tuple shape, e.g. from stream_mode="custom" when
            # multiple stream modes are combined internally by this langgraph version.
            _, payload = chunk
            out.append(payload)
        else:
            out.append(chunk)
    return out


async def start(graph, cfg, conversation_id="t1") -> list[dict]:
    initial = MatterDraft(conversation_id=conversation_id, correlation_id=conversation_id)
    return await collect(graph.astream(initial, cfg, stream_mode="custom"))


async def send(graph, cfg, payload: dict) -> list[dict]:
    return await collect(graph.astream(Command(resume=payload), cfg, stream_mode="custom"))


def cards(evs: list[dict]) -> list[str]:
    return [e["card"] for e in evs if e.get("type") == "card"]


def last_card(evs: list[dict], name: str) -> dict:
    return next(e["props"] for e in reversed(evs) if e.get("type") == "card" and e["card"] == name)


def errors(evs: list[dict]) -> list[str]:
    return [e["code"] for e in evs if e.get("type") == "error"]
