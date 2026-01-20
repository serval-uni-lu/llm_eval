import json

from llm_eval_framework.metrics.base import BaseMetric, MetricResult


class IsJsonMetric(BaseMetric):
    """Metric that checks if output is valid JSON."""

    def __init__(self, schema: dict = None):
        """Initialize IsJsonMetric.

        Args:
            schema: Optional JSON schema to validate against (not implemented yet)
        """
        super().__init__(name="is_json")
        self.schema = schema

    def score(self, output: str) -> MetricResult:
        """Check if output is valid JSON.

        Args:
            output: The text to check

        Returns:
            MetricResult with value 1.0 if valid JSON, 0.0 otherwise
        """
        try:
            parsed = json.loads(output)
            # TODO: Add schema validation if self.schema is provided
            return MetricResult(
                value=1.0,
                details=f"Valid JSON object with {len(parsed) if isinstance(parsed, (dict, list)) else 'scalar'} items",
            )
        except (json.JSONDecodeError, ValueError) as e:
            return MetricResult(value=0.0, details=f"Invalid JSON: {str(e)}")
