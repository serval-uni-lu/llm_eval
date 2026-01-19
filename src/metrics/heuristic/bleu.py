from typing import Union, List

from src.metrics.base import BaseMetric, MetricResult

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None


class BLEUMetric(BaseMetric):
    """Metric that calculates BLEU score using NLTK."""

    def __init__(self, n_grams: int = 4, smoothing: str = "method1", case_sensitive: bool = False):
        """Initialize BLEUMetric.

        Args:
            n_grams: Maximum n-gram order to use (default 4 for BLEU-4)
            smoothing: Smoothing method name from NLTK (default "method1")
            case_sensitive: Whether comparison should be case-sensitive
        """
        super().__init__(name="bleu")

        if sentence_bleu is None:
            raise ImportError("NLTK required for BLEU. Install with: pip install nltk")

        self.n_grams = n_grams
        self.case_sensitive = case_sensitive
        self.weights = tuple(1.0 / n_grams for _ in range(n_grams))
        self.smoothing_func = getattr(SmoothingFunction(), smoothing, SmoothingFunction().method1)

    def score(self, output: str, reference: Union[str, List[str]]) -> MetricResult:
        """Calculate BLEU score between output and reference(s).

        Args:
            output: The generated text to evaluate
            reference: Reference text(s) - can be a single string or list of strings

        Returns:
            MetricResult with BLEU score (0.0 to 1.0)
        """
        if not output.strip():
            return MetricResult(value=0.0, details={"error": "Empty output"})

        output_normalized = self._normalize_text(output, self.case_sensitive)
        candidate = output_normalized.split()

        # Handle single or multiple references
        if isinstance(reference, str):
            if not reference.strip():
                return MetricResult(value=0.0, details={"error": "Empty reference"})
            references = [self._normalize_text(reference, self.case_sensitive).split()]
        else:
            references = []
            for ref in reference:
                if ref.strip():
                    references.append(self._normalize_text(ref, self.case_sensitive).split())
            if not references:
                return MetricResult(value=0.0, details={"error": "No valid references"})

        try:
            bleu_score = sentence_bleu(
                references,
                candidate,
                weights=self.weights,
                smoothing_function=self.smoothing_func
            )
            return MetricResult(
                value=bleu_score,
                details={
                    "n_grams": self.n_grams,
                    "case_sensitive": self.case_sensitive
                }
            )
        except ZeroDivisionError:
            return MetricResult(value=0.0, details={"error": "ZeroDivisionError in BLEU calculation"})
