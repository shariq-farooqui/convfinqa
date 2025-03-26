# In llm/client.py
from abc import ABC, abstractmethod

from convfinqa.dataset.models import LLMResponse


class LLMClient(ABC):
    """Abstract base class for large language model clients."""

    @abstractmethod
    async def generate_response(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> LLMResponse:
        """Generate a response from the language model.

        Args:
            system_prompt: System instructions for the model
            user_prompt: User query or input
            **kwargs: Additional model-specific parameters

        Returns:
            Pydantic model containing response and metadata
        """
        pass
