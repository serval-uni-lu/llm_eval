import json
import yaml
from pathlib import Path

from llm_eval_framework.llm import LLM
from llm_eval_framework.dataset import Dataset
from llm_eval_framework.decorators import retry
from llm_eval_framework.metrics import Metric, list_metrics
from llm_eval_framework.utils import normalize_text, ensure_dir, clear_cuda_cache
from llm_eval_framework.basemodels import ModelConfig, DatasetConfig, EvaluationConfig


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

        # Load LLM
        llm = LLM(
            model_name=model_config.name,
            load_in_4bit=model_config.load_in_4bit,
            load_in_8bit=model_config.load_in_8bit,
        )

        # Extract generation parameters
        sampling_params = {}
        if model_config.sampling_params:
            sampling_params = model_config.sampling_params.model_dump(exclude_none=True)

        for dataset_config in config.datasets:
            print(
                f"  on dataset: {dataset_config.name} with metrics: {', '.join(dataset_config.metrics)}"
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
            for i, prompt in enumerate(dataset.prompts):
                llm_output = llm.generate(prompt, **sampling_params)

                # Get row data from dataset
                row = dataset.data.iloc[i].to_dict()

                output = {
                    "prompt": prompt,
                    "response": llm_output.content,
                    **row,  # Include all dataset fields (query, context, etc.)
                }
                outputs.append(output)
                print(f"    [{i + 1}/{len(dataset)}] Generated response")

            # Save outputs
            with open(output_path, "w") as f:
                for output in outputs:
                    f.write(json.dumps(output) + "\n")

            print(f"  Saved {len(outputs)} outputs to {output_path}")

        # Unload LLM and clear cache
        llm.unload()
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
    for model_config in config.models:
        for dataset_config in config.datasets:
            heuristic_to_compute = [
                m for m in dataset_config.metrics if m in heuristic_metrics
            ]
            if not heuristic_to_compute:
                continue

            print(
                f"\nComputing heuristic metrics for model={model_config.name}, dataset={dataset_config.name}"
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

    if judge_needed:
        if not config.judge_model:
            print("\nWarning: LLM judge metrics requested but no judge_model provided")
            return

        # Clear CUDA cache before loading judge model
        clear_cuda_cache()

        print(f"\nLoading LLM judge: {config.judge_model.name}")
        llm_judge = LLM(
            model_name=config.judge_model.name,
            load_in_4bit=config.judge_model.load_in_4bit,
            load_in_8bit=config.judge_model.load_in_8bit,
            enable_compile=False,
        )

        # Extract sampling params from judge config
        judge_sampling_params = {}
        if config.judge_model.sampling_params:
            judge_sampling_params = config.judge_model.sampling_params.model_dump(
                exclude_none=True
            )

        for model_config in config.models:
            for dataset_config in config.datasets:
                judge_to_compute = [
                    m for m in dataset_config.metrics if m in llm_judge_metrics
                ]
                if not judge_to_compute:
                    continue

                print(
                    f"\nComputing LLM judge metrics for model={model_config.name}, dataset={dataset_config.name}"
                )

                outputs, output_dir = _load_outputs(
                    config, model_config, dataset_config
                )
                if not outputs:
                    continue

                for metric_name in judge_to_compute:
                    _compute_judge_metric(
                        metric_name,
                        outputs,
                        output_dir,
                        llm_judge,
                        judge_sampling_params,
                    )

        print("\nUnloading LLM judge...")
        llm_judge.unload()


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
    metric = Metric(metric_name)

    results = []
    for output_data in outputs:
        result = metric.score(output_data["response"], output_data.get("answer"))

        results.append(
            {
                "score": result.value,
                "details": result.details,
            }
        )

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
    llm_judge: LLM,
    sampling_params: dict = None,
):
    """Compute an LLM judge metric for all outputs."""

    # Check if metric already computed
    metrics_dir = output_dir / "metrics"
    metric_path = metrics_dir / f"{metric_name}.jsonl"
    if metric_path.exists():
        print(f"  ✓ Metric already computed, skipping: {metric_name}")
        return

    print(f"  Computing LLM judge metric: {metric_name}")
    metric = Metric(metric_name)

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def score_with_retry(llm, input_text, output_text, reference, sampling_params):
        return metric.score(
            llm, input_text, output_text, reference, sampling_params=sampling_params
        )

    results = []
    for i, output_data in enumerate(outputs):
        try:
            result = score_with_retry(
                llm_judge,
                output_data["prompt"],
                output_data["response"],
                output_data.get("answer"),
                sampling_params,
            )

            results.append(
                {
                    "score": result.value,
                    "details": result.details,
                }
            )
            print(f"    [{i + 1}/{len(outputs)}] Score: {result.value:.3f}")

            clear_cuda_cache()

        except Exception as e:
            # Graceful error handling after all retries failed
            error_result = {
                "score": None,
                "details": None,
                "error": str(e).split("\n")[0],
                "error_type": type(e).__name__,
                "prompt": output_data["prompt"],
                "response": output_data["response"],
            }

            # Include enriched error context if available
            if hasattr(e, "eval_prompt"):
                error_result["eval_prompt"] = e.eval_prompt
            if hasattr(e, "judge_output"):
                error_result["judge_model_output"] = e.judge_output

            results.append(error_result)
            print(
                f"    [{i + 1}/{len(outputs)}] ERROR: {str(e).split(chr(10))[0][:80]}"
            )

            clear_cuda_cache()

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
            f"    Saved {len(results)} results ({len(successful_scores)} successful, avg: {avg_score:.3f})"
        )
    else:
        print(f"    Saved {len(results)} results (all failed)")

