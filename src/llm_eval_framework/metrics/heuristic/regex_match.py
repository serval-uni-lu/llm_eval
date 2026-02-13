import re
from typing import Union

from ..base import BaseMetric, MetricResult


class RegexMatchMetric(BaseMetric):
    """Metric that checks if output matches a regular expression pattern."""

    def __init__(self, pattern: Union[str, re.Pattern] = None):
        """Initialize RegexMatchMetric.

        Args:
            pattern: Regex pattern (string or compiled) to match against (can also be passed in score())
        """
        super().__init__(name="regex_match")
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    def score(
        self, output: str, pattern: Union[str, re.Pattern] = None
    ) -> MetricResult:
        """Check if output matches the regex pattern.

        Args:
            output: The text to check
            pattern: Regex pattern to match (overrides constructor arg if provided)

        Returns:
            MetricResult with value 1.0 if matches, 0.0 otherwise
        """
        pattern_to_use = pattern if pattern is not None else self.pattern

        if pattern_to_use is None:
            raise ValueError(
                "Pattern must be provided either in constructor or score()"
            )

        # Compile if string
        if isinstance(pattern_to_use, str):
            pattern_to_use = re.compile(pattern_to_use)

        matches = pattern_to_use.search(output) is not None

        return MetricResult(
            value=1.0 if matches else 0.0,
            details=f"Pattern {'matched' if matches else 'did not match'} in output",
        )
