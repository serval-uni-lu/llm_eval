import inspect
from typing import Optional
from dataclasses import dataclass

from src.llm import LLM
from src.metrics.base import MetricResult


@dataclass
class ParsedArgs:
    """Parsed metric arguments with semantic meaning."""
    output: Optional[str] = None
    input: Optional[str] = None
    reference: Optional[str] = None
    llm: Optional[LLM] = None


class Metric:
    """Smart wrapper providing unified API for all metrics.

    Usage:
        metric = Metric("is_json")
        metric.score(output)

        metric = Metric("contains")
        metric.score(output, reference)

        metric = Metric("instruction_following")
        metric.score(input, output, llm)
        metric.score(input, output, reference, llm)
    """

    def __init__(self, name: str, **config):
        """Initialize metric wrapper.

        Args:
            name: Metric name from registry
            **config: Metric-specific configuration
        """
        from src.metrics.loader import get_metric, load_registry

        self._name = name
        self._metric = get_metric(name, **config)
        self._type = self._detect_type(load_registry())
        self._signature = inspect.signature(self._metric.score)

    def _detect_type(self, registry: dict) -> str:
        """Detect if this is a judge or heuristic metric."""
        if self._name in registry.get('llm_judge', {}):
            return 'judge'
        return 'heuristic'

    def score(self, *args, **kwargs) -> MetricResult:
        """Flexible scoring API.

        Accepts positional or keyword arguments and automatically maps
        to the native metric signature.

        Returns:
            MetricResult with score and details
        """
        parsed = self._parse_positional(args, kwargs)
        return self._call_with_parsed(parsed, kwargs)

    def _parse_positional(self, args: tuple, kwargs: dict) -> ParsedArgs:
        """Parse positional arguments into semantic structure."""
        llm = None
        strings = []

        for arg in args:
            if arg is None:
                continue
            elif isinstance(arg, LLM):
                llm = arg
            elif isinstance(arg, str):
                strings.append(arg)
            else:
                raise TypeError(f"Unexpected argument type: {type(arg)}")

        parsed = ParsedArgs(llm=llm)

        if self._type == 'judge':
            if len(strings) >= 1:
                parsed.input = strings[0]
            if len(strings) >= 2:
                parsed.output = strings[1]
            if len(strings) >= 3:
                parsed.reference = strings[2]
        else:
            if len(strings) >= 1:
                parsed.output = strings[0]
            if len(strings) >= 2:
                parsed.reference = strings[1]

        for key, value in kwargs.items():
            if key == 'input':
                parsed.input = value
            elif key == 'output':
                parsed.output = value
            elif key in ('reference', 'substring', 'expected', 'pattern'):
                parsed.reference = value
            elif key == 'llm':
                parsed.llm = value

        return parsed

    def _call_with_parsed(self, parsed: ParsedArgs, original_kwargs: dict = None) -> MetricResult:
        """Map parsed args to native signature and call metric."""
        call_kwargs = {}
        if original_kwargs is None:
            original_kwargs = {}

        # Known parameter names that are handled by ParsedArgs
        known_params = {'output', 'input', 'reference', 'llm', 'substring', 'expected', 'pattern'}

        for param_name, param in self._signature.parameters.items():
            if param_name == 'self':
                continue

            if param_name == 'output' and parsed.output is not None:
                call_kwargs['output'] = parsed.output
            elif param_name == 'input' and parsed.input is not None:
                call_kwargs['input'] = parsed.input
            elif param_name == 'reference' and parsed.reference is not None:
                call_kwargs['reference'] = parsed.reference
            elif param_name == 'llm' and parsed.llm is not None:
                call_kwargs['llm'] = parsed.llm
            elif param_name == 'substring' and parsed.reference is not None:
                call_kwargs['substring'] = parsed.reference
            elif param_name == 'expected' and parsed.reference is not None:
                call_kwargs['expected'] = parsed.reference
            elif param_name == 'pattern' and param.default != inspect.Parameter.empty:
                continue
            elif param.default != inspect.Parameter.empty:
                continue
            elif param_name not in call_kwargs:
                raise ValueError(
                    f"Missing required parameter '{param_name}' for metric '{self._name}'"
                )

        # Pass through any extra kwargs not in known_params
        for key, value in original_kwargs.items():
            if key not in known_params:
                call_kwargs[key] = value

        return self._metric.score(**call_kwargs)
