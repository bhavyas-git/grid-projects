from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def _approximate_section(text: str) -> str:
    for line in text.splitlines():
        candidate = " ".join(line.split()).strip()
        if not candidate:
            continue
        if len(candidate) < 4:
            continue
        if candidate.isupper() or candidate.startswith("SECTION "):
            return candidate
        return candidate[:80]
    return "Unknown"


def load_and_chunk(pdf_path) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    for page_number, document in enumerate(documents, start=1):
        document.metadata["page"] = document.metadata.get("page", page_number - 1)
        document.metadata["page_number"] = page_number
        document.metadata["source_type"] = "text"
        document.metadata["source_path"] = str(pdf_path)
        document.metadata["source_name"] = pdf_path.name
        document.metadata["approximate_section"] = _approximate_section(document.page_content)
        document.metadata["content_type"] = "text"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)
    for chunk_id, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["source_type"] = "text"
        chunk.metadata["content_type"] = "text"
        chunk.metadata["character_count"] = len(chunk.page_content)
    return chunks
