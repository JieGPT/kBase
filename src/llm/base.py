from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMResponse:
    def __init__(self, content: str, tokens_used: int, model: str):
        self.content = content
        self.tokens_used = tokens_used
        self.model = model


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
