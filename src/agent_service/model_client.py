from typing import AsyncIterator, Protocol

import anthropic
from pydantic import BaseModel

from .config import Settings


class ModelClient(Protocol):
    async def extract(self, schema: type[BaseModel], instruction: str,
                      text: str) -> BaseModel | None: ...
    def converse_stream(self, system: str, user: str) -> AsyncIterator[str]: ...


class AnthropicModelClient:
    """Extractor: Haiku-class structured outputs. Conversationalist: Sonnet-class streaming."""

    def __init__(self, settings: Settings):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        self.extractor_model = settings.extractor_model
        self.converse_model = settings.converse_model

    async def extract(self, schema, instruction, text):
        try:
            resp = await self.client.messages.parse(
                model=self.extractor_model,
                max_tokens=1024,
                system=instruction,
                messages=[{"role": "user", "content": text}],
                output_format=schema,
            )
            return resp.parsed_output
        except anthropic.APIError:
            return None

    async def converse_stream(self, system, user):
        async with self.client.messages.stream(
            model=self.converse_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for delta in stream.text_stream:
                yield delta


class StubModelClient:
    """Deterministic client for tests and USE_STUB_MODEL mode."""

    def __init__(self, extractions: list[BaseModel] | None = None, reply: str = "Okay."):
        self.extractions = list(extractions or [])
        self.reply = reply

    async def extract(self, schema, instruction, text):
        return self.extractions.pop(0) if self.extractions else None

    async def converse_stream(self, system, user):
        yield self.reply


def make_model_client(settings: Settings) -> ModelClient:
    return StubModelClient() if settings.use_stub_model else AnthropicModelClient(settings)
