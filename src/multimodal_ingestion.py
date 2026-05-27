import base64
import json
from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config import (
    IMAGE_OUTPUT_DIR,
    MAX_PAGE_CONTEXT_CHARS,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    PDF_PATH,
    TABLE_OUTPUT_DIR,
)
from src.llm import get_chat_model


class ImageDescription(BaseModel):
    caption: str = Field(description="A concise description of the image or chart.")
    chart_type: str | None = Field(
        default=None,
        description="The likely chart or image type, if applicable.",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Important entities, metrics, or labels visible in the image.",
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _ensure_output_dirs() -> None:
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _page_context(page: fitz.Page) -> str:
    return _normalize_text(page.get_text("text"))[:MAX_PAGE_CONTEXT_CHARS]


def _caption_image(image_bytes: bytes, page_number: int) -> ImageDescription:
    model = get_chat_model().with_structured_output(ImageDescription)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Describe this image from page {page_number} of the IFC annual report. "
                    "Focus on finance-relevant content, visible labels, and any trends."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
    )
    return model.invoke([message])


def extract_image_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    _ensure_output_dirs()
    pdf = fitz.open(pdf_path)
    image_docs: list[Document] = []
    manifest: list[dict] = []

    for page_index, page in enumerate(pdf, start=1):
        page_context = _page_context(page)
        seen_xrefs: set[int] = set()
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            extracted = pdf.extract_image(xref)
            width = extracted.get("width", 0)
            height = extracted.get("height", 0)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            image_bytes = extracted["image"]
            image_ext = extracted.get("ext", "png")
            image_path = IMAGE_OUTPUT_DIR / f"page_{page_index:03d}_image_{image_index:02d}.{image_ext}"
            image_path.write_bytes(image_bytes)

            try:
                description = _caption_image(image_bytes, page_index)
            except Exception:
                description = ImageDescription(
                    caption="Image extracted from the report. Caption generation failed.",
                    chart_type=None,
                    key_entities=[],
                )

            metadata = {
                "source": str(pdf_path),
                "source_name": pdf_path.name,
                "page": page_index - 1,
                "page_number": page_index,
                "source_type": "image",
                "content_type": "image",
                "image_path": str(image_path),
                "image_index": image_index,
                "width": width,
                "height": height,
                "chart_type": description.chart_type,
                "key_entities": description.key_entities,
                "page_context": page_context,
                "approximate_section": page_context[:80] or "Unknown",
            }
            manifest.append(metadata)
            image_docs.append(
                Document(
                    page_content=(
                        f"Image caption: {description.caption}\n"
                        f"Chart type: {description.chart_type or 'unknown'}\n"
                        f"Key entities: {', '.join(description.key_entities) or 'none'}\n"
                        f"Page context: {page_context}"
                    ),
                    metadata=metadata,
                )
            )

    (IMAGE_OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return image_docs


def _table_to_records(rows: list[list[str | None]]) -> tuple[list[str], list[dict[str, str]], str]:
    sanitized_rows = [[(cell or "").strip() for cell in row] for row in rows if any(cell for cell in row)]
    if not sanitized_rows:
        return [], [], ""

    header = sanitized_rows[0]
    normalized_header: list[str] = []
    for index, cell in enumerate(header, start=1):
        normalized_header.append(cell or f"column_{index}")

    body_rows = sanitized_rows[1:] or sanitized_rows
    records = [
        {
            normalized_header[column_index]: row[column_index] if column_index < len(row) else ""
            for column_index in range(len(normalized_header))
        }
        for row in body_rows
    ]
    if records:
        separator = "| " + " | ".join(["---"] * len(normalized_header)) + " |"
        header_row = "| " + " | ".join(normalized_header) + " |"
        body = [
            "| " + " | ".join(record.get(column, "") for column in normalized_header) + " |"
            for record in records
        ]
        markdown = "\n".join([header_row, separator, *body])
    else:
        markdown = ""
    return normalized_header, records, markdown


def extract_table_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    _ensure_output_dirs()
    pdf = fitz.open(pdf_path)
    table_docs: list[Document] = []
    manifest: list[dict] = []

    for page_index, page in enumerate(pdf, start=1):
        page_context = _page_context(page)
        try:
            tables = page.find_tables()
        except Exception:
            continue

        for table_index, table in enumerate(tables.tables, start=1):
            rows = table.extract()
            headers, records, markdown = _table_to_records(rows)
            if not records:
                continue

            json_path = TABLE_OUTPUT_DIR / f"page_{page_index:03d}_table_{table_index:02d}.json"
            md_path = TABLE_OUTPUT_DIR / f"page_{page_index:03d}_table_{table_index:02d}.md"
            csv_path = TABLE_OUTPUT_DIR / f"page_{page_index:03d}_table_{table_index:02d}.csv"
            json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")
            csv_lines = [",".join(headers)] + [
                ",".join(json.dumps(record.get(column, "")) for column in headers)
                for record in records
            ]
            csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

            metadata = {
                "source": str(pdf_path),
                "source_name": pdf_path.name,
                "page": page_index - 1,
                "page_number": page_index,
                "source_type": "table",
                "content_type": "table",
                "table_index": table_index,
                "table_json_path": str(json_path),
                "table_markdown_path": str(md_path),
                "table_csv_path": str(csv_path),
                "columns": headers,
                "column_count": len(headers),
                "row_count": len(records),
                "page_context": page_context,
                "approximate_section": page_context[:80] or "Unknown",
            }
            manifest.append(metadata)
            table_docs.append(
                Document(
                    page_content=(
                        f"Table extracted from page {page_index}.\n"
                        f"Columns: {', '.join(headers)}\n"
                        f"Rows: {len(records)}\n"
                        f"{markdown}\n\n"
                        f"Page context: {page_context}"
                    ),
                    metadata=metadata,
                )
            )

    (TABLE_OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return table_docs


def load_all_documents(
    text_documents: list[Document],
    include_images: bool = True,
    include_tables: bool = True,
    pdf_path: Path = PDF_PATH,
) -> list[Document]:
    documents = list(text_documents)
    if include_images:
        documents.extend(extract_image_documents(pdf_path))
    if include_tables:
        documents.extend(extract_table_documents(pdf_path))
    return documents
