import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ..base import BaseMetric, MetricResult


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
            MetricResult with compound sentiment score (0 to 1.0)
            - 0.5 indicates neutral sentiment
            - Values less than 0.5 indicate negative sentiment
            - Values higher than 0.5 indicate positive sentiment
        """
        if not output.strip():
            return MetricResult(value=0.0, details={"error": "Empty output"})

        scores = self.analyzer.polarity_scores(output)
        # normalise `compound` score from (-1, 1) to (0, 1)
        compound_score = (scores["compound"] + 1) / 2

        return MetricResult(
            value=compound_score,
            details={
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "compound": compound_score,
            },
        )
