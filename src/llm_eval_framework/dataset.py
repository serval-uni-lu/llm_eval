import yaml
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from .prompt import Prompt


class DatasetMetadata(BaseModel):
    name: str
    description: str
    task: str


class Dataset:
    def __init__(self, data, metadata, prompt_template):
        """Initialize the Dataset with data, metadata, and prompt template.

        Args:
            data (pd.DataFrame): The dataset in tabular form.
            metadata (DatasetMetadata): Metadata about the dataset.
            prompt_template (Prompt): The prompt template for generating prompts.
        """
        self.data = data
        self.metadata = metadata
        self.prompt_template = prompt_template

    @staticmethod
    def from_path(path: str | Path) -> "Dataset":
        """Load dataset from the specified directory path."""
        path = Path(path)
        data_file = path / "dataset.parquet"
        metadata_file = path / "metadata.json"
        prompt_file = path / "prompt.yaml"

        # data
        data = pd.read_parquet(data_file)

        # metadata
        with open(metadata_file, "r") as f:
            metadata_dict = yaml.safe_load(f)
            metadata = DatasetMetadata(**metadata_dict)

        # prompt
        prompt = Prompt.from_file(prompt_file)

        return Dataset(data, metadata, prompt)

    @property
    def prompts(self) -> list[str]:
        """Return the list of prompts generated from the dataset."""
        prompts = []
        fields = self.prompt_template.fields()
        for _, row in self.data.iterrows():
            kwargs = {}
            for field in fields:
                if field in row:
                    kwargs[field] = row[field]
                elif field == "choices":
                    choice_cols = [
                        col for col in self.data.columns if col.startswith("choice")
                    ]
                    choices = [
                        f"{chr(65 + i)}. {row[col]}"
                        for i, col in enumerate(sorted(choice_cols))
                        if col in row
                    ]
                    kwargs["choices"] = "\n".join(choices)
            prompts.append(self.prompt_template.format(**kwargs))
        return prompts

    @property
    def answers(self) -> list[str]:
        """Return the list of answers if available."""
        if "answer" in self.data.columns:
            return self.data["answer"].tolist()
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
