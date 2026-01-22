import math
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from ..base import BaseMetric, MetricResult
from ...prompt import Prompt
from ...llm import LLM, LLMOutput


class GEvalOutput(BaseModel):
    """Validated output format for G-Eval metric."""

    score: int = Field(..., ge=0, le=5, description="Score from 0 to 5")
    reason: str = Field(..., min_length=10, description="Reasoning for the score")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Ensure reason is a non-empty string with meaningful content."""
        if not v or not v.strip():
            raise ValueError("Reason must be a non-empty string")
        return v.strip()


class GEval(BaseMetric):
    """G-Eval metric using chain-of-thought prompting for LLM-based evaluation."""

    def __init__(
        self,
        name: str,
        task_introduction: str,
        evaluation_criteria: str,
        chain_of_thought: str,
    ):
        """Initialize G-Eval metric.

        Args:
            name: Name of the metric
            task_introduction: Task description
            evaluation_criteria: Evaluation criteria
            chain_of_thought: Chain of thought prompt
        """
        super().__init__(name=name)
        self.task_introduction = task_introduction
        self.evaluation_criteria = evaluation_criteria
        self.chain_of_thought = chain_of_thought

        template_path = Path(__file__).parent / "templates/g_eval.yaml"
        with open(template_path) as f:
            self.prompt_templates = yaml.safe_load(f)

    def score(
        self,
        llm: LLM,
        input: str,
        output: str,
        reference: Optional[str] = None,
        sampling_params: dict = None,
    ) -> MetricResult:
        """Compute the G-Eval metric score using LLM judge.

        Args:
            llm: LLM instance to use as judge
            input: The input prompt
            output: The model output to evaluate
            reference: Optional reference answer
            sampling_params: Sampling parameters for generation (max_new_tokens, temperature, etc.)

        Returns:
            MetricResult with normalized score and reasoning details
        """
        if sampling_params is None:
            sampling_params = {"max_new_tokens": 512}
            # sampling_params = {}

        # Prepare template fields
        fields = {
            "task_introduction": self.task_introduction,
            "evaluation_criteria": self.evaluation_criteria,
            "chain_of_thought": self.chain_of_thought,
            "input": input,
            "output": output,
        }

        # Select appropriate template
        if reference:
            prompt_template = Prompt(self.prompt_templates["with_reference"])
            fields["reference"] = reference
        else:
            prompt_template = Prompt(self.prompt_templates["no_reference"])

        eval_prompt = prompt_template.format(**fields)

        # Generate evaluation with logprobs and sampling params
        raw_output = llm.generate(eval_prompt, topk_logprobs=10, **sampling_params)
        # print(f"Judge Output: {raw_output.content.replace('\n', ' ')}")

        # Parse and return result
        try:
            return self.parse_model_output(raw_output)
        except ValueError as e:
            # Enrich error with context for debugging
            error_msg = str(e)
            error_msg += f"\n\nEval Prompt:\n{eval_prompt[:500]}..."
            error_msg += f"\n\nJudge Output:\n{repr(raw_output.content)}"

            # Store context in exception for access in evaluation pipeline
            enriched_error = ValueError(error_msg)
            enriched_error.eval_prompt = eval_prompt
            enriched_error.judge_output = raw_output.content
            raise enriched_error

    def parse_model_output(self, output: LLMOutput) -> MetricResult:
        """Parse model output and return weighted score with reasoning details.

        Args:
            output: LLMOutput from the judge model

        Returns:
            MetricResult with normalized score and reasoning
        """
        output_json = output.extract_json()
        if output_json is None:
            raise ValueError(
                f"No valid JSON output found in output. Content: {repr(output.content[:200])}"
            )

        # Validate output format with Pydantic
        try:
            validated_output = GEvalOutput(**output_json)
        except Exception as e:
            raise ValueError(f"Invalid output format: {e}. Got: {output_json}")

        # Try to compute weighted score from logprobs
        try:
            score = self.compute_weighted_score(output)
        except (ValueError, KeyError) as e:
            # Fallback to validated score if logprobs computation fails
            print(f"WARNING: Logprobs computation failed: {e}. Using extracted score.")
            score = float(validated_output.score)

        normalized_score = score / 5.0  # normalize to [0, 1] range

        return MetricResult(value=normalized_score, details=validated_output.reason)

    def compute_weighted_score(
        self, output: LLMOutput, score_tokens: list = None
    ) -> float:
        """
        Compute weighted score using token probabilities from logprobs.

        Args:
            output: LLMOutput containing logprobs
            score_tokens: Valid score tokens (defaults to ['0', '1', '2', '3', '4', '5'])

        Returns:
            Weighted score based on token probabilities.
        """
        if not output.logprobs or not output.logprobs.get("content"):
            raise ValueError("No logprobs available in output")

        if score_tokens is None:
            score_tokens = ["0", "1", "2", "3", "4", "5"]

        content_logprobs = output.logprobs["content"]

        # Find the position where the score token appears
        score_token_position = None
        for i, token_info in enumerate(content_logprobs):
            token = token_info["token"].strip()
            if token in score_tokens:
                score_token_position = i
                break

        if score_token_position is None:
            try:
                output_json = output.extract_json()
                score = float(output_json.get("score"))
                raise ValueError(
                    f"No valid score token found in logprobs. Extracted score: {score}"
                )
            except Exception as e:
                raise ValueError(f"Error extracting score: {e}")

        # Get probabilities for all possible score tokens at that position
        token_info = content_logprobs[score_token_position]
        all_token_probs = [token_info] + token_info.get("top_logprobs", [])

        score_probs = {}
        for prob_info in all_token_probs:
            token = prob_info["token"].strip()
            if token in score_tokens:
                score_value = int(token)
                probability = math.exp(prob_info["logprob"])
                score_probs[score_value] = probability

        if not score_probs:
            try:
                output_json = output.extract_json()
                score = float(output_json.get("score"))
                raise ValueError(
                    f"No valid score probabilities found. Extracted score: {score}"
                )
            except Exception as e:
                raise ValueError(f"Error extracting score: {e}")

        # Compute weighted score
        weighted_score = sum(score * prob for score, prob in score_probs.items())

        return weighted_score
