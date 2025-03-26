import os

from openai import APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from convfinqa.dataset.models import LLMResponse

from .client import LLMClient


class OpenAIClient(LLMClient):
    """Asynchronous client for OpenAI LLMs."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI()

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
    )
    async def generate_response(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate a response from the language model."""
        response = await self._client.beta.chat.completions.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format=LLMResponse,
        )
        return response
