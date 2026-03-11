import pytest
from src.llm.openai_client import OpenAIClient


@pytest.mark.asyncio
async def test_generate():
    llm = OpenAIClient(model="gpt-4o-mini")
    response = await llm.generate("Say 'hello'")
    assert response.content
    assert response.tokens_used > 0


@pytest.mark.asyncio
async def test_stream():
    llm = OpenAIClient(model="gpt-4o-mini")
    chunks = []
    async for chunk in llm.stream("Say 'hello'"):
        chunks.append(chunk)
    assert "".join(chunks)
