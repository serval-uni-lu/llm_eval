from src.metrics.base import BaseMetric, MetricResult


class EqualsMetric(BaseMetric):
    """Metric that checks if output exactly equals an expected value."""

    def __init__(self, expected: str = None, case_sensitive: bool = True, strip: bool = True):
        """Initialize EqualsMetric.

        Args:
            expected: The expected value (can also be passed in score())
            case_sensitive: Whether comparison should be case-sensitive
            strip: Whether to strip whitespace before comparison
        """
        super().__init__(name="equals")
        self.expected = expected
        self.case_sensitive = case_sensitive
        self.strip = strip

    def score(self, output: str, expected: str = None) -> MetricResult:
        """Check if output equals expected value.

        Args:
            output: The text to check
            expected: The expected value (overrides constructor arg if provided)

        Returns:
            MetricResult with value 1.0 if equal, 0.0 otherwise
        """
        expected_value = expected if expected is not None else self.expected

        if expected_value is None:
            raise ValueError("Expected value must be provided either in constructor or score()")

        # Process strings
        output_processed = output.strip() if self.strip else output
        expected_processed = expected_value.strip() if self.strip else expected_value

        if not self.case_sensitive:
            output_processed = output_processed.lower()
            expected_processed = expected_processed.lower()

        matches = output_processed == expected_processed

        return MetricResult(
            value=1.0 if matches else 0.0,
            details=f"Output {'matches' if matches else 'does not match'} expected value"
        )
