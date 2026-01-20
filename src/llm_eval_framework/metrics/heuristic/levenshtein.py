from rapidfuzz.distance import Indel

from llm_eval_framework.metrics.base import BaseMetric, MetricResult


class LevenshteinMetric(BaseMetric):
    """Metric that calculates normalized Levenshtein (edit distance) similarity."""

    def __init__(self, case_sensitive: bool = False):
        """Initialize LevenshteinMetric.

        Args:
            case_sensitive: Whether comparison should be case-sensitive
        """
        super().__init__(name="levenshtein")

        if Indel is None:
            raise ImportError(
                "rapidfuzz required for Levenshtein. Install with: pip install rapidfuzz"
            )

        self.case_sensitive = case_sensitive

    def score(self, output: str, reference: str) -> MetricResult:
        """Calculate normalized Levenshtein similarity between output and reference.

        Args:
            output: The text to evaluate
            reference: The reference text to compare against

        Returns:
            MetricResult with similarity score (0.0 to 1.0)
        """
        output_normalized = self._normalize_text(output, self.case_sensitive)
        reference_normalized = self._normalize_text(reference, self.case_sensitive)

        similarity = Indel.normalized_similarity(
            output_normalized, reference_normalized
        )

        return MetricResult(
            value=similarity, details={"case_sensitive": self.case_sensitive}
        )
