import re
from pathlib import Path


class Prompt:

    def __init__(self, template: str):
        """Initialize a Prompt with a template string.

        Args:
            template: A string containing {{field_name}} placeholders
        """
        self.template = template

    @staticmethod
    def from_file(file_path: str | Path):
        """Load a prompt template from a file.

        Args:
            file_path: Path to the template file

        Returns:
            Prompt instance with the file contents as template
        """
        path = Path(file_path)

        if path.suffix in ['.yaml', '.yml']:
            import yaml
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            template = data['template']
        else:
            with path.open('r', encoding='utf-8') as f:
                template = f.read()

        return Prompt(template)

    def fields(self) -> list[str]:
        """Extract all field names from the template.

        Returns a list of input fields found in {{field_name}} placeholders.
        """
        # Find all {{field_name}} patterns
        pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(pattern, self.template)
        # Return unique field names in order of appearance
        seen = set()
        result = []
        for field in matches:
            if field not in seen:
                seen.add(field)
                result.append(field)
        return result

    def format(self, **fields) -> str:
        """Format the template by replacing {{field_name}} with provided values.

        Args:
            **fields: Keyword arguments where keys match field names in template

        Returns:
            Formatted string with all placeholders replaced
        """
        result = self.template
        for field_name, value in fields.items():
            placeholder = f'{{{{{field_name}}}}}'
            result = result.replace(placeholder, str(value))
        return result
