import os
from typing import AsyncIterator, Optional
from openai import AsyncOpenAI
import tiktoken
from .base import BaseLLM, LLMResponse


class OpenAIClient(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Use provided values or fall back to legacy env var for backward compatibility
        api_key_str: str = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
        base_url_str: Optional[str] = base_url or os.getenv("BASE_URL")

        if not api_key_str:
            raise ValueError(
                "API key is required. Set API_KEY or OPENAI_API_KEY environment variable."
            )

        if base_url_str:
            self.client = AsyncOpenAI(api_key=api_key_str, base_url=base_url_str)
        else:
            self.client = AsyncOpenAI(api_key=api_key_str)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._encoding = tiktoken.encoding_for_model(model)

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model=self.model,
        )

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    async def generate_with_messages(self, messages: list, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        return LLMResponse(
            content=content,
            tokens_used=tokens_used,
            model=self.model,
        )
