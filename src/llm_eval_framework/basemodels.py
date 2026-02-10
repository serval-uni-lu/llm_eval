import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional


class SamplingParams(BaseModel):
    """Sampling parameters for model generation."""

    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str
    sampling_params: Optional[SamplingParams] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    batch_size: int = 1


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str
    metrics: List[str]


class MetricConfig(BaseModel):
    """Metric configuration."""

    name: str
    weight: float = 1.0


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    name: str
    models: List[ModelConfig]
    datasets: List[DatasetConfig]
    judge_model: Optional[ModelConfig] = None

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "EvaluationConfig":
        """Load evaluation config from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return EvaluationConfig(**data)
