"""Pydantic models for the ConvFinQA dataset."""

import json
from typing import Literal

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    """Model for storing reasoning steps taken for an LLM response."""

    description: str = Field(
        ..., description="Explanation of this reasoning step"
    )
    reference: str | None = Field(
        None, description="Reference to specific data being used"
    )


class LLMResponse(BaseModel):
    """Model for storing structured LLM responses."""

    steps: list[ReasoningStep] = Field(
        ..., description="Step-by-step reasoning process without calculations"
    )
    expression: str = Field(
        ..., description="Mathematical expression that can be evaluated"
    )


class QASample(BaseModel):
    """A single question-answer sample with context."""

    id: str
    question: str = ""
    answer: str | None = None
    pre_text: str
    post_text: str
    table: list[list[str]]
    exe_ans: float | str | None = None
    llm_response: LLMResponse | None = None
    predicted_answer: float | None = None
    accurate: bool | None = None

    @property
    def is_valid(self) -> bool:
        """Check if both question and answer are present."""
        return (
            bool(self.question)
            and self.answer is not None
            and self.answer != ""
        )

    @property
    def answer_type(
        self,
    ) -> Literal[
        "numeric",
        "clean_percentage",
        "text_with_percentage",
        "currency",
        "boolean",
        "other",
        "invalid",
    ]:
        """Determine the type of answer."""
        if not self.is_valid:
            return "invalid"

        answer_str = str(self.answer).strip()

        # Check for boolean answers
        if answer_str.lower() in ["yes", "no", "true", "false"]:
            return "boolean"

        # Check for clean percentage (number followed by %)
        if answer_str.endswith("%"):
            # Remove the % and try to convert to float
            numeric_part = answer_str[:-1].strip()
            try:
                float(numeric_part.replace(",", ""))
                return "clean_percentage"  # It's a number followed by %
            except ValueError:
                return (
                    "text_with_percentage"  # Contains % but not a clean number
                )

        # Check for currency answers
        currency_indicators = [
            "$",
            "€",
            "£",
            "¥",
            "usd",
            "eur",
            "gbp",
            "jpy",
            "dollar",
            "euro",
            "pound",
            "yen",
        ]
        if any(
            indicator in answer_str.lower() for indicator in currency_indicators
        ):
            return "currency"

        # Check if purely numeric
        try:
            float(answer_str.replace(",", ""))
            return "numeric"
        except ValueError:
            return "other"

    @property
    def formatted_context(self) -> str:
        """Formats the context from a sample for the API request."""
        table_str = json.dumps(self.table)

        context = f"""
            ## Pre-text\n{self.pre_text}\n
            ## Table\n{table_str}\n
            ## Post-text: {self.post_text}\n

            Question: {self.question}
        """
        return context

    @property
    def decimal_precision(self) -> int:
        """Determine the number of decimal places in the answer."""
        if not self.is_valid:
            return 0

        # If this is a percentage and exe_ans is available, use precision from exe_ans
        if self.answer_type == "clean_percentage" and self.exe_ans is not None:
            # Convert exe_ans to string and check its decimal precision
            exe_ans_str = str(self.exe_ans)
            if "." in exe_ans_str:
                return len(exe_ans_str.split(".")[-1])
            return 0

        answer_str = str(self.answer).strip()

        # Handle percentage answers without exe_ans
        if self.answer_type == "clean_percentage":
            # Remove the % and any commas, then check decimal part
            numeric_part = answer_str[:-1].strip().replace(",", "")
            if "." in numeric_part:
                return len(numeric_part.split(".")[-1])
            return 0

        # Handle pure numeric answers
        elif self.answer_type == "numeric":
            # Remove commas
            answer_str = answer_str.replace(",", "").strip()
            if "." in answer_str:
                return len(answer_str.split(".")[-1])

        return 0


class Dataset(BaseModel):
    """The complete ConvFinQA dataset as a collection of QA samples."""

    samples: list[QASample]

    @property
    def valid_samples(self) -> list[QASample]:
        """Get all valid samples in the dataset."""
        return [sample for sample in self.samples if sample.is_valid]
