from pydantic import BaseModel

from agent_service.model_client import AnthropicModelClient, StubModelClient, make_model_client


class Toy(BaseModel):
    name: str | None = None


async def test_stub_extract_scripted():
    stub = StubModelClient(extractions=[Toy(name="hello")])
    out = await stub.extract(Toy, "extract", "some text")
    assert out.name == "hello"
    assert await stub.extract(Toy, "extract", "again") is None


async def test_stub_converse_stream():
    stub = StubModelClient(reply="Hi there.")
    chunks = [c async for c in stub.converse_stream("sys", "user")]
    assert "".join(chunks) == "Hi there."


def test_factory_returns_stub(monkeypatch):
    from agent_service import config
    s = config.Settings(use_stub_model=True)
    assert isinstance(make_model_client(s), StubModelClient)


def test_live_client_has_bounded_timeout_and_retries(monkeypatch):
    from agent_service import model_client
    from agent_service.config import Settings

    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_client.anthropic, "AsyncAnthropic", FakeAnthropic)
    AnthropicModelClient(Settings(anthropic_api_key="test"))
    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 2
