import json
import yaml
from pathlib import Path

from .base import BaseMetric


def load_registry() -> dict:
    """Load metric registry from JSON file.

    Returns:
        Dictionary with 'heuristic' and 'llm_judge' sections
    """
    registry_path = Path(__file__).parent / "registry.json"
    with open(registry_path) as f:
        return json.load(f)


def get_metric(name: str, **kwargs) -> BaseMetric:
    """Factory function to create metric instances.

    Args:
        name: Name of the metric
        **kwargs: Additional arguments passed to metric constructor

    Returns:
        BaseMetric instance
    """
    registry = load_registry()

    if name in registry.get("heuristic", {}):
        return _load_heuristic_metric(name, registry["heuristic"][name], kwargs)

    if name in registry.get("llm_judge", {}):
        return _load_llm_judge_metric(name, registry["llm_judge"][name], kwargs)

    available_metrics = list(registry.get("heuristic", {}).keys()) + list(
        registry.get("llm_judge", {}).keys()
    )
    raise ValueError(
        f"Metric '{name}' not found in registry. "
        f"Available metrics: {', '.join(available_metrics)}"
    )


def _load_heuristic_metric(name: str, config: dict, kwargs: dict) -> BaseMetric:
    """Load a heuristic metric by importing its class.

    Args:
        name: Metric name
        config: Registry config for this metric
        kwargs: Constructor arguments

    Returns:
        BaseMetric instance
    """
    from . import heuristic

    # Import the class
    metric_class = getattr(heuristic, config["class"])

    # Instantiate with provided kwargs
    return metric_class(**kwargs)


def _load_llm_judge_metric(name: str, config: dict, kwargs: dict) -> BaseMetric:
    """Load an LLM judge metric from template.

    Args:
        name: Metric name
        config: Registry config for this metric
        kwargs: Constructor arguments (ignored for template-based metrics)

    Returns:
        BaseMetric instance (GEval)
    """
    from .llm_judge.g_eval import GEval

    # Load template configuration
    template_name = config["template"]
    templates_dir = Path(__file__).parent / "llm_judge" / "templates"
    template_path = templates_dir / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path) as f:
        template_config = yaml.safe_load(f)

    # Create GEval instance with template config
    return GEval(
        name=name,
        task_introduction=template_config["task_introduction"],
        evaluation_criteria=template_config["evaluation_criteria"],
        chain_of_thought=template_config["chain_of_thought"],
    )


def list_metrics() -> dict:
    """List all available metrics.

    Returns:
        Dictionary with 'heuristic' and 'llm_judge' keys containing metric info
    """
    registry = load_registry()

    return {
        "heuristic": {
            name: config.get("description", "No description")
            for name, config in registry.get("heuristic", {}).items()
        },
        "llm_judge": {
            name: config.get("description", "No description")
            for name, config in registry.get("llm_judge", {}).items()
        },
    }
