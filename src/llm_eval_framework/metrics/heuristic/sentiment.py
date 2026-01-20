import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from llm_eval_framework.metrics.base import BaseMetric, MetricResult


class SentimentMetric(BaseMetric):
    """Metric that analyzes sentiment using NLTK's VADER."""

    def __init__(self):
        """Initialize SentimentMetric."""
        super().__init__(name="sentiment")

        if SentimentIntensityAnalyzer is None:
            raise ImportError(
                "NLTK required for sentiment. Install with: pip install nltk"
            )

        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()

    def score(self, output: str) -> MetricResult:
        """Analyze sentiment polarity of the output text.

        Args:
            output: The text to analyze

        Returns:
            MetricResult with compound sentiment score (-1.0 to 1.0)
            - Negative values indicate negative sentiment
            - Positive values indicate positive sentiment
            - Zero indicates neutral sentiment
        """
        if not output.strip():
            return MetricResult(value=0.0, details={"error": "Empty output"})

        scores = self.analyzer.polarity_scores(output)
        compound_score = scores["compound"]

        return MetricResult(
            value=compound_score,
            details={
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "compound": compound_score,
            },
        )
