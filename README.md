# LLM Evaluation Framework

A framework for evaluating LLMs with support for multiple metrics, datasets, and models.

## Installation

```bash
uv pip install "git+https://github.com/serval-uni-lu/llm_eval"
```

## Usage

### LLM Generation

```python
from llm_eval_framework.llm import LLM

llm = LLM('google/gemma-3-4b-it')
output = llm.generate("Your prompt here", temperature=0.7, top_p=0.9)
print(output.content)
llm.unload()
```

### Dataset

Datasets require three files: `data.parquet`, `metadata.json`, and `prompt.yaml`.

```python
from llm_eval_framework.dataset import Dataset

dataset = Dataset.from_path('path/to/dataset')
print(dataset.prompts[0])
print(dataset.answers[0])
```

### Metrics

Supports heuristic metrics (`is_json`, `contains`, `bleu`, `rouge`, etc.) and LLM-judge metrics (`answer_correctness`, `bias`, `safety`, etc.).

```python
from llm_eval_framework.llm import LLM
from llm_eval_framework.metrics import Metric

llm = LLM('meta-llama/Llama-3.2-3B-Instruct')

correctness = Metric("answer_correctness")
score = correctness.score(input, output, reference, llm)

is_json = Metric("is_json")
score = is_json.score(output)

llm.unload()
```

### Evaluation

Configure evaluations via YAML:

```yaml
name: experiment_name

models:
- name: Qwen/Qwen3-4B-Instruct-2507
  sampling_params:
    temperature: 0.7

datasets:
- name: financebench
  metrics:
  - answer_correctness

judge_model:
  name: google/gemma-3-4b-it
  sampling_params:
    temperature: 0.4
```

Run evaluation:

```python
from llm_eval_framework.evaluation import EvaluationConfig, run_evaluation

config = EvaluationConfig.from_yaml("eval_config.yaml")
results = run_evaluation(config)
```

### Document Parsing

Parse PDFs to markdown using Docling:

```python
from llm_eval_framework.parser import Parser

parser = Parser()
markdown_files = parser.parse('input_dir', 'output_dir')
```

### Semantic Chunking

```python
from llm_eval_framework.chunker import Chunker

chunker = Chunker()
chunks = chunker.chunk(text)
```

### Visualization

```python
from llm_eval_framework.visualization import save_results_plot

save_results_plot(output_dir='path/to/results', save_path='plot.png')
```
