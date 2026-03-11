"""LLM client implementation using LangChain."""

from typing import AsyncIterator, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from .base import BaseLLM, LLMResponse


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect tokens as they're generated."""
        self.tokens.append(token)


class OpenAIClient(BaseLLM):
    """OpenAI-compatible LLM client using LangChain."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the OpenAI client.

        Args:
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: Optional base URL for API
            api_key: Optional API key
        """
        # Build kwargs for ChatOpenAI
        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = ChatOpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and metadata
        """
        # Create message
        message = HumanMessage(content=prompt)

        # Call LLM
        response = await self.client.ainvoke([message])

        # Get token usage if available
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        return LLMResponse(
            content=response.content,
            tokens_used=tokens_used,
            model=self.model,
        )

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream a response from the LLM.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        # Create message
        message = HumanMessage(content=prompt)

        # Stream response
        async for chunk in self.client.astream([message]):
            if chunk.content:
                yield chunk.content

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        # Use LangChain's token counting
        try:
            return self.client.get_num_tokens(text)
        except Exception:
            # Fallback: rough estimate (approx 4 chars per token)
            return len(text) // 4

    async def generate_with_messages(self, messages: list, **kwargs) -> LLMResponse:
        """Generate response from list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and metadata
        """
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
        )

        # Convert dict messages to LangChain messages
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Call LLM
        response = await self.client.ainvoke(lc_messages)

        # Get token usage if available
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        return LLMResponse(
            content=response.content,
            tokens_used=tokens_used,
            model=self.model,
        )
