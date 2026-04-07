import json
import yaml
from functools import partial
from itertools import product
from pathlib import Path

from .llm import LLMGenerationWrapper
from .dataset import Dataset
from .decorators import retry_batches
from .metrics import list_metrics
from .metrics.loader import get_metric
from .utils import get_items, normalize_text, ensure_dir, clear_cuda_cache
from .basemodels import ModelConfig, DatasetConfig, EvaluationConfig

try:
    # NOTE: Added in python 3.12
    from itertools import batched
except ImportError:
    from .utils import batched


def run_evaluation(config: EvaluationConfig):
    """Run evaluation for the specified configuration."""
    output_dir = Path(f"data/outputs/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
    print(f"Saved evaluation config to {config_path}")

    generate_outputs(config)
    compute_metrics(config)


def generate_outputs(config: EvaluationConfig):
    """Generate outputs for all model/dataset combinations."""
    for model_config in config.models:
        print(f"Evaluating model: {model_config.name}")
        model_norm = normalize_text(model_config.name)

        # Check if outputs already exist for this model
        all_exist = True
        for dataset_config in config.datasets:
            dataset_norm = normalize_text(dataset_config.name)
            output_path = Path(
                f"data/outputs/{config.name}/{dataset_norm}/{model_norm}/outputs.jsonl"
            )
            if not output_path.exists():
                all_exist = False
                break

        if all_exist:
            print("  ✓ All outputs already exist for this model, skipping entirely")
            continue

        # Extract generation parameters
        sampling_params = {}
        if model_config.sampling_params:
            sampling_params = model_config.sampling_params.model_dump(exclude_none=True)

        llm_generation_wrapper = LLMGenerationWrapper(
            model_name=model_config.name,
            load_in_4bit=model_config.load_in_4bit,
            load_in_8bit=model_config.load_in_8bit,
            sampling_params=sampling_params,
            endpoint=config.endpoint,
        )

        for dataset_config in config.datasets:
            print(
                f"  on dataset: {dataset_config.name} with metrics: "
                f"{', '.join(dataset_config.metrics)}"
            )

            # Load dataset
            dataset = Dataset.from_path(f"data/datasets/{dataset_config.name}")
            dataset_norm = normalize_text(dataset_config.name)

            # Create output path
            output_dir = Path(f"data/outputs/{config.name}/{dataset_norm}/{model_norm}")
            ensure_dir(output_dir)
            output_path = output_dir / "outputs.jsonl"

            # Skip if outputs already exist
            if output_path.exists():
                print(f"  ✓ Outputs already exist, skipping generation: {output_path}")
                continue

            # Generate outputs
            outputs = []
            for batch in dataset.iter(model_config.batch_size):
                # NOTE: batch := indices, rows, prompts, answers
                llm_outputs = llm_generation_wrapper.generate(batch[2])

                for i, row, prompt, answer, llm_output in zip(*batch, llm_outputs):
                    output = row.copy()
                    output.update(
                        dict(
                            prompt=prompt,
                            answer=answer,
                            response=llm_output.content,
                        )
                    )
                    print(f"    [{i + 1}/{len(dataset)}] Generated response")
                    outputs.append(output)

            # Save outputs
            with open(output_path, "w") as f:
                for output in outputs:
                    f.write(json.dumps(output) + "\n")

            print(f"  Saved {len(outputs)} outputs to {output_path}")

        # Unload LLM and clear cache
        llm_generation_wrapper.unload()
        clear_cuda_cache()


def compute_metrics(config: EvaluationConfig):
    """Compute metrics for all generated outputs.

    Two-phase approach:
    1. Compute all heuristic metrics (fast, no LLM needed)
    2. Load judge once, compute all LLM-judge metrics, unload
    """
    metric_types = list_metrics()
    heuristic_metrics = set(metric_types["heuristic"].keys())
    llm_judge_metrics = set(metric_types["llm_judge"].keys())

    # Phase 1: Heuristic metrics
    for model_config, dataset_config in product(config.models, config.datasets):
        heuristic_to_compute = [
            m for m in dataset_config.metrics if m in heuristic_metrics
        ]
        if not heuristic_to_compute:
            continue

        print(
            f"\nComputing heuristic metrics for model={model_config.name}, "
            f"dataset={dataset_config.name}"
        )

        outputs, output_dir = _load_outputs(config, model_config, dataset_config)
        if not outputs:
            continue

        for metric_name in heuristic_to_compute:
            _compute_heuristic_metric(metric_name, outputs, output_dir)

    # Phase 2: LLM judge metrics
    judge_needed = any(
        any(m in llm_judge_metrics for m in dataset.metrics)
        for dataset in config.datasets
    )

    if not judge_needed or not config.judge_model:
        print("\nWarning: LLM judge metrics requested but no judge_model provided")
        return

    # Clear CUDA cache before loading judge model
    clear_cuda_cache()

    # Extract sampling params from judge config
    judge_sampling_params = {}
    if config.judge_model.sampling_params:
        judge_sampling_params = config.judge_model.sampling_params.model_dump(
            exclude_none=True
        )

    print(f"\nLoading LLM judge: {config.judge_model.name}")
    llm_judge_generation_wrapper = LLMGenerationWrapper(
        model_name=config.judge_model.name,
        load_in_4bit=config.judge_model.load_in_4bit,
        load_in_8bit=config.judge_model.load_in_8bit,
        sampling_params=judge_sampling_params,
        endpoint=config.endpoint,
    )

    for model_config, dataset_config in product(config.models, config.datasets):
        judge_to_compute = [m for m in dataset_config.metrics if m in llm_judge_metrics]
        if not judge_to_compute:
            continue

        print(
            f"\nComputing LLM judge metrics for model={model_config.name}, "
            f"dataset={dataset_config.name}"
        )

        outputs, output_dir = _load_outputs(config, model_config, dataset_config)
        if not outputs:
            continue

        for metric_name in judge_to_compute:
            _compute_judge_metric(
                metric_name,
                outputs,
                output_dir,
                llm_judge_generation_wrapper,
            )

    print("\nUnloading LLM judge...")
    llm_judge_generation_wrapper.unload()


def compute_metrics_in_batches(
    outputs,
    retries,
    batch_size,
    metric_name,
    llm_judge_generation_wrapper: LLMGenerationWrapper | None = None,
    return_all_scores: bool = False,
    **metric_kwargs,
) -> list[dict]:
    metric = get_metric(metric_name, **metric_kwargs)

    metric_types = list_metrics()
    heuristic_metrics = set(metric_types["heuristic"].keys())
    is_heuristic = metric_name in heuristic_metrics

    _compute_batched = (
        partial(_compute_batched_heuristic_metric, metric=metric)
        if is_heuristic
        else partial(
            _compute_batched_judge_metric,
            metric=metric,
            llm_judge_generation_wrapper=llm_judge_generation_wrapper,
        )
    )

    @retry_batches(retries=retries)
    def _process_batches(llm_inputs: list[dict], batch_size: int):
        """
        Processes one round of batches.
        Returns:
            results: dict{sub_index -> parsed_output}
            failed: dict{sub_index -> enriched error str}
        """
        results = {}
        failed = {}

        for sub_indices in batched(range(len(llm_inputs)), batch_size):
            batch = get_items(llm_inputs, *sub_indices, batch=False)
            batch_results = _compute_batched(batch=batch)

            for i, result in zip(sub_indices, batch_results):
                if result.get("score") is None:
                    failed[i] = result
                else:
                    results[i] = result

        return results, failed

    all_results = _process_batches(outputs, batch_size)

    if return_all_scores:
        return all_results

    valid_results = []

    for i, result in enumerate(all_results, start=1):
        if result.get("score") is None:
            error_msg = result.get("error", "")[:80]
            if error_msg:
                print(f"    [{i}/{len(outputs)}] ERROR: {error_msg}")
        else:
            valid_results.append(result)

    return valid_results


def _compute_batched_heuristic_metric(metric, batch, **kwargs) -> list[dict]:
    batch_results = []
    for output in batch:
        try:
            result_metric = metric.score(output.get("response"), output.get("answer"))
            result = {"score": result_metric.value, "details": result_metric.details}
        except Exception as e:
            result = {
                "score": None,
                "details": None,
                "error": str(e).split("\n")[0],
                "error_type": type(e).__name__,
            }
        batch_results.append(result)

    return batch_results


def _compute_batched_judge_metric(
    metric, batch, llm_judge_generation_wrapper
) -> list[dict]:
    prompts = [
        metric.prepare_eval_prompt(
            output.get("prompt"), output.get("response"), output.get("answer")
        )
        for output in batch
    ]

    raw_outputs = llm_judge_generation_wrapper.generate(prompts)

    batch_results = []
    for raw_output, output in zip(raw_outputs, batch):
        try:
            result_metric = metric.parse_model_output(raw_output)
            result = {"score": result_metric.value, "details": result_metric.details}
        except Exception as e:
            # Graceful error handling after all retries failed
            result = {
                "score": None,
                "details": None,
                "error": str(e).split("\n")[0],
                "error_type": type(e).__name__,
                "prompt": output.get("prompt"),
                "response": output.get("response"),
            }

            # Include enriched error context if available
            if hasattr(e, "eval_prompt"):
                result["eval_prompt"] = e.eval_prompt
            if hasattr(e, "judge_output"):
                result["judge_model_output"] = e.judge_output

        batch_results.append(result)

    return batch_results


def _load_outputs(
    config: EvaluationConfig, model_config: ModelConfig, dataset_config: DatasetConfig
):
    """Load outputs for a model/dataset combination."""
    model_norm = normalize_text(model_config.name)
    dataset_norm = normalize_text(dataset_config.name)
    output_dir = Path(f"data/outputs/{config.name}/{dataset_norm}/{model_norm}")
    outputs_path = output_dir / "outputs.jsonl"

    if not outputs_path.exists():
        print(f"  Warning: Outputs not found at {outputs_path}")
        return None, output_dir

    outputs = []
    with open(outputs_path, "r") as f:
        for line in f:
            outputs.append(json.loads(line))

    print(f"  Loaded {len(outputs)} outputs")
    return outputs, output_dir


def _compute_heuristic_metric(metric_name: str, outputs: list, output_dir: Path):
    """Compute a heuristic metric for all outputs."""

    # Check if metric already computed
    metrics_dir = output_dir / "metrics"
    metric_path = metrics_dir / f"{metric_name}.jsonl"
    if metric_path.exists():
        print(f"  ✓ Metric already computed, skipping: {metric_name}")
        return

    print(f"  Computing heuristic metric: {metric_name}")

    results = compute_metrics_in_batches(outputs, 1, 1, metric_name)

    # Save to file
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metric_path = metrics_dir / f"{metric_name}.jsonl"
    with open(metric_path, "w") as f:
        for result_data in results:
            f.write(json.dumps(result_data) + "\n")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"    Saved {len(results)} results (avg: {avg_score:.3f})")


def _compute_judge_metric(
    metric_name: str,
    outputs: list,
    output_dir: Path,
    llm_judge_generation_wrapper: LLMGenerationWrapper,
):
    """Compute an LLM judge metric for all outputs."""

    # Check if metric already computed
    metrics_dir = output_dir / "metrics"
    metric_path = metrics_dir / f"{metric_name}.jsonl"
    if metric_path.exists():
        print(f"  ✓ Metric already computed, skipping: {metric_name}")
        return

    print(f"  Computing LLM judge metric: {metric_name}")

    results = compute_metrics_in_batches(
        outputs,
        retries=3,
        batch_size=10,
        metric_name=metric_name,
        llm_judge_generation_wrapper=llm_judge_generation_wrapper,
    )

    # Save to file
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metric_path = metrics_dir / f"{metric_name}.jsonl"
    with open(metric_path, "w") as f:
        for result_data in results:
            f.write(json.dumps(result_data) + "\n")

    # Calculate average only for successful scores
    successful_scores = [r["score"] for r in results if r["score"] is not None]
    if successful_scores:
        avg_score = sum(successful_scores) / len(successful_scores)
        print(
            f"    Saved {len(results)} results ({len(successful_scores)} successful,"
            f"avg: {avg_score:.3f})"
        )
    else:
        print(f"    Saved {len(results)} results (all failed)")
