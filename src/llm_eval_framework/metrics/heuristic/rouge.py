from typing import Union, List
from rouge_score import rouge_scorer

from ..base import BaseMetric, MetricResult


class ROUGEMetric(BaseMetric):
    """Metric that calculates ROUGE score using rouge-score library."""

    def __init__(self, rouge_type: str = "rouge1", use_stemmer: bool = False):
        """Initialize ROUGEMetric.

        Args:
            rouge_type: Type of ROUGE to calculate (rouge1, rouge2, rougeL, rougeLsum)
            use_stemmer: Whether to use stemming for matching
        """
        super().__init__(name="rouge")

        if rouge_scorer is None:
            raise ImportError(
                "rouge-score required for ROUGE. Install with: pip install rouge-score"
            )

        valid_types = {"rouge1", "rouge2", "rougeL", "rougeLsum"}
        if rouge_type not in valid_types:
            raise ValueError(
                f"Invalid rouge_type '{rouge_type}'. Must be one of {valid_types}"
            )

        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)
        self.use_stemmer = use_stemmer

    def score(self, output: str, reference: Union[str, List[str]]) -> MetricResult:
        """Calculate ROUGE score between output and reference(s).

        Args:
            output: The generated text to evaluate
            reference: Reference text(s) - can be a single string or list of strings

        Returns:
            MetricResult with ROUGE F-measure score (0.0 to 1.0)
        """
        if not output.strip():
            return MetricResult(value=0.0, details={"error": "Empty output"})

        # Handle single or multiple references
        if isinstance(reference, str):
            if not reference.strip():
                return MetricResult(value=0.0, details={"error": "Empty reference"})
            references = [reference]
        else:
            references = [ref for ref in reference if ref.strip()]
            if not references:
                return MetricResult(value=0.0, details={"error": "No valid references"})

        # Use score_multi for multiple references
        results = self.scorer.score_multi(references, output)
        value = results[self.rouge_type].fmeasure

        return MetricResult(
            value=value,
            details={
                "rouge_type": self.rouge_type,
                "use_stemmer": self.use_stemmer,
            },
        )
