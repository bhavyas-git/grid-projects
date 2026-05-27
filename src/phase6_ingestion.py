import base64
import json
from pathlib import Path

import fitz
import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config import (
    PAGE_IMAGE_DIR,
    PATCH_OUTPUT_DIR,
    PDF_PATH,
    PHASE6_PATCH_CONTEXT_CHARS,
    PHASE6_PATCH_EMBEDDINGS_PATH,
    PHASE6_PATCH_MIN_TEXT_CHARS,
    PHASE6_PATCH_PADDING,
    PHASE6_PATCH_STORE_PATH,
    PHASE6_PATCH_TOKEN_EMBEDDINGS_PATH,
)
from src.embeddings import get_embeddings
from src.llm import get_chat_model


class PatchDescription(BaseModel):
    summary: str = Field(description="Concise summary of the visual patch.")
    visible_text: str | None = Field(
        default=None,
        description="Short visible text or labels in the patch, if any.",
    )
    visual_signals: list[str] = Field(
        default_factory=list,
        description="Important visual elements, metrics, or chart cues.",
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _ensure_output_dirs() -> None:
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE6_PATCH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _page_context(page: fitz.Page) -> str:
    return _normalize_text(page.get_text("text"))[:PHASE6_PATCH_CONTEXT_CHARS]


def _safe_filename(page_number: int, patch_index: int) -> str:
    return f"page_{page_number:03d}_patch_{patch_index:03d}.png"


def _expanded_rect(rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    expanded = fitz.Rect(
        max(page_rect.x0, rect.x0 - PHASE6_PATCH_PADDING),
        max(page_rect.y0, rect.y0 - PHASE6_PATCH_PADDING),
        min(page_rect.x1, rect.x1 + PHASE6_PATCH_PADDING),
        min(page_rect.y1, rect.y1 + PHASE6_PATCH_PADDING),
    )
    return expanded


def _render_page_image(page: fitz.Page, page_image_path: Path) -> None:
    if page_image_path.exists():
        return
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    pixmap.save(page_image_path)


def _describe_patch(image_bytes: bytes, page_number: int, patch_index: int) -> PatchDescription:
    model = get_chat_model().with_structured_output(PatchDescription)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Describe this document patch from page {page_number} of the IFC annual report. "
                    f"This is patch {patch_index}. Focus on finance-relevant values, labels, table cues, and visual structure."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
    )
    return model.invoke([message])


def _iter_candidate_blocks(page: fitz.Page):
    for block_index, block in enumerate(page.get_text("blocks"), start=1):
        if len(block) < 7:
            continue
        x0, y0, x1, y1, text, _, block_type = block[:7]
        rect = fitz.Rect(x0, y0, x1, y1)
        normalized_text = _normalize_text(text or "")
        if rect.width < 40 or rect.height < 24:
            continue
        if block_type == 0 and len(normalized_text) < PHASE6_PATCH_MIN_TEXT_CHARS:
            continue
        if block_type not in {0, 1}:
            continue
        yield block_index, rect, normalized_text, block_type


def _patch_interaction_units(
    *,
    summary: str,
    visible_text: str | None,
    visual_signals: list[str],
    page_context: str,
    patch_role: str,
) -> list[str]:
    units = [
        f"Patch role: {patch_role}",
        f"Patch summary: {summary}",
    ]
    if visible_text:
        units.append(f"Visible text: {visible_text}")
    units.extend(f"Visual signal: {signal}" for signal in visual_signals if signal)
    if page_context:
        units.append(f"Page context: {page_context}")
    return [unit for unit in units if unit.strip()]


def build_phase6_patch_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    _ensure_output_dirs()
    pdf = fitz.open(pdf_path)
    patch_documents: list[Document] = []
    manifest: list[dict] = []
    interaction_units_by_patch: list[list[str]] = []

    for page_number, page in enumerate(pdf, start=1):
        page_context = _page_context(page)
        page_image_path = PAGE_IMAGE_DIR / f"page_{page_number:03d}.png"
        _render_page_image(page, page_image_path)

        patch_index = 0
        for block_number, rect, block_text, block_type in _iter_candidate_blocks(page):
            patch_index += 1
            clip_rect = _expanded_rect(rect, page.rect)
            patch_path = PATCH_OUTPUT_DIR / _safe_filename(page_number, patch_index)
            if not patch_path.exists():
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect, alpha=False)
                pixmap.save(patch_path)

            patch_bytes = patch_path.read_bytes()
            patch_role = "text_patch" if block_type == 0 else "image_patch"
            summary = "Text-heavy document patch."
            visible_text = block_text or None
            visual_signals: list[str] = []

            if block_type == 1 or len(block_text) < 180 or any(
                token in block_text.lower() for token in ("figure", "chart", "graph", "income", "assets", "fy24", "fy23")
            ):
                try:
                    description = _describe_patch(patch_bytes, page_number, patch_index)
                    summary = description.summary
                    visible_text = description.visible_text or visible_text
                    visual_signals = description.visual_signals
                except Exception:
                    summary = "Document patch extracted from the report."

            metadata = {
                "source": str(pdf_path),
                "source_name": pdf_path.name,
                "page": page_number - 1,
                "page_number": page_number,
                "source_type": "visual_patch",
                "content_type": "visual_patch",
                "patch_index": patch_index,
                "block_index": block_number,
                "patch_path": str(patch_path),
                "page_image_path": str(page_image_path),
                "bbox": [round(clip_rect.x0, 2), round(clip_rect.y0, 2), round(clip_rect.x1, 2), round(clip_rect.y1, 2)],
                "patch_role": patch_role,
                "visible_text": visible_text or "",
                "visual_signals": visual_signals,
                "page_context": page_context,
                "approximate_section": (block_text[:80] or page_context[:80] or "Visual patch"),
            }
            content = (
                f"Patch summary: {summary}\n"
                f"Visible text: {visible_text or 'none'}\n"
                f"Visual signals: {', '.join(visual_signals) or 'none'}\n"
                f"Page context: {page_context}\n"
                f"Patch role: {patch_role}"
            )
            interaction_units = _patch_interaction_units(
                summary=summary,
                visible_text=visible_text,
                visual_signals=visual_signals,
                page_context=page_context,
                patch_role=patch_role,
            )
            manifest.append({"page_content": content, "metadata": metadata})
            interaction_units_by_patch.append(interaction_units)
            patch_documents.append(Document(page_content=content, metadata=metadata))

    pdf.close()
    PHASE6_PATCH_STORE_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    embeddings = get_embeddings()
    patch_vectors = embeddings.embed_documents([document.page_content for document in patch_documents]) if patch_documents else []
    np.save(PHASE6_PATCH_EMBEDDINGS_PATH, np.array(patch_vectors, dtype=np.float32))
    flat_units = [unit for patch_units in interaction_units_by_patch for unit in patch_units]
    flat_vectors = embeddings.embed_documents(flat_units) if flat_units else []
    token_records: list[dict] = []
    cursor = 0
    for patch_units in interaction_units_by_patch:
        vectors = flat_vectors[cursor : cursor + len(patch_units)]
        token_records.append(
            {
                "units": patch_units,
                "vectors": vectors,
            }
        )
        cursor += len(patch_units)
    PHASE6_PATCH_TOKEN_EMBEDDINGS_PATH.write_text(json.dumps(token_records), encoding="utf-8")

    from src.vectorstore import create_qdrant_vectorstore

    # after embeddings are created
    vectorstore = create_qdrant_vectorstore(
        documents=patch_documents,
        embeddings=embeddings,
        collection_name="ifc_annual_report_2024_phase6_patches",  # IMPORTANT
    )
    return patch_documents

if __name__ == "__main__":
    print("Running Phase 6 ingestion...")
    docs = build_phase6_patch_documents()
    print(f"Created {len(docs)} patch documents and stored in Qdrant.")
