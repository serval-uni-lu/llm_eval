from typing import Optional, Any
from pydantic import BaseModel, Field

class MetricResult(BaseModel):
    """Result from metric computation."""
    value: float = Field(..., description="Metric score value")
    details: Optional[Any] = Field(None, description="Additional details about the result")
    
    def __str__(self):
        return f" value={self.value}\n details={self.details}\n"
    
class BaseMetric:
    """Base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    def score(self, *args, **kwargs) -> MetricResult:
        """Compute the metric score.

        Subclasses should define their own signature based on requirements:
        - Simple metrics: score(output)
        - Reference-based: score(output, reference)
        - Context-aware: score(input, output, reference=None)
        - LLM-judged: score(llm, input, output, reference=None)

        Returns:
            MetricResult with score and details
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def _normalize_text(text: str, case_sensitive: bool = True) -> str:
        """Normalize text for comparison.

        Args:
            text: Input text to normalize
            case_sensitive: If False, convert to lowercase

        Returns:
            Normalized text with whitespace stripped
        """
        normalized = text.strip()
        if not case_sensitive:
            normalized = normalized.lower()
        return normalized
