from src.metrics.base import BaseMetric, MetricResult


class ContainsMetric(BaseMetric):
    """Metric that checks if output contains a specific substring."""

    def __init__(self, substring: str = None, case_sensitive: bool = True):
        """Initialize ContainsMetric.

        Args:
            substring: The substring to search for (can also be passed in score())
            case_sensitive: Whether the search should be case-sensitive
        """
        super().__init__(name="contains")
        self.substring = substring
        self.case_sensitive = case_sensitive

    def score(self, output: str, substring: str = None) -> MetricResult:
        """Check if output contains the substring.

        Args:
            output: The text to check
            substring: The substring to search for (overrides constructor arg if provided)

        Returns:
            MetricResult with value 1.0 if found, 0.0 otherwise
        """
        substring_to_check = substring if substring is not None else self.substring

        if substring_to_check is None:
            raise ValueError("Substring must be provided either in constructor or score()")

        if not self.case_sensitive:
            found = substring_to_check.lower() in output.lower()
        else:
            found = substring_to_check in output

        return MetricResult(
            value=1.0 if found else 0.0,
            details=f"Substring '{substring_to_check}' {'found' if found else 'not found'} in output"
        )
