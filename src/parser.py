from pathlib import Path
from loguru import logger
from typing import List, Union
from docling.document_converter import DocumentConverter


class Parser:
    """Simple document parser using Docling."""

    def __init__(self, device: str = "auto"):
        """Initialize parser.

        Args:
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = device
        self.converter = None

    def _get_converter(self):
        """Lazy initialization of DocumentConverter."""
        if self.converter is None:
            self.converter = DocumentConverter()
            logger.info(f"Initialized Docling converter")
        return self.converter

    def parse(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        remove_image_tags: bool = True,
    ) -> List[str]:
        """Parse all PDFs in input_dir to markdown files in output_dir.

        Args:
            input_dir: Directory containing PDF documents
            output_dir: Directory to save markdown files

        Returns:
            List of created markdown file contents
        """
        docs_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pdfs = list(docs_path.glob("*.pdf"))

        if not pdfs:
            logger.warning(f"No PDFs found in {input_dir}")
            return []

        logger.info(f"Found {len(pdfs)} PDF(s) to parse")

        converter = self._get_converter()
        md_contents = []

        for i, pdf in enumerate(pdfs, 1):
            logger.info(f"[{i}/{len(pdfs)}] Processing {pdf.name}")

            md_path = output_path / f"{pdf.stem}.md"

            if md_path.exists():
                logger.info(f"Skipping {md_path.name}, already exists - reading cached version")
                markdown = md_path.read_text(encoding='utf-8')
                md_contents.append(markdown)
                continue
            
            try:
                result = converter.convert(str(pdf))
                markdown = result.document.export_to_markdown()
                
                if remove_image_tags:
                    markdown = markdown.replace("\n<!-- image -->", "")

                md_path.write_text(markdown, encoding='utf-8')
                md_contents.append(markdown)

                logger.info(f"Saved {md_path.name}")

            except Exception as e:
                logger.error(f"Failed {pdf.name}: {e}")

        logger.info(f"Completed: {len(md_contents)}/{len(pdfs)} successful")
        return md_contents
