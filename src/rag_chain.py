from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL


def _prompt_for_style(answer_style: str = "default") -> PromptTemplate:
    style_instructions = (
        "Give a slightly detailed answer in 2 to 4 sentences.\n"
        "Lead with the direct answer, then add the most relevant supporting detail.\n"
    )
    if answer_style == "table":
        style_instructions = (
            "Return the answer as a compact markdown table whenever the question asks for values, comparisons, or fiscal-year breakdowns.\n"
            "Use a short lead-in sentence only if needed.\n"
            "If two or more periods or categories are compared, the markdown table must include one row per period/category.\n"
            "Do not compress numeric answers into prose if a table is appropriate.\n"
        )
    if answer_style == "image":
        style_instructions = (
            "Prioritize image-derived and chart-derived evidence from the provided context.\n"
            "Use numeric values from chart labels or adjacent page context when they are explicitly present.\n"
            "Answer directly and mention that the value comes from the relevant figure when appropriate.\n"
        )
    if answer_style == "hybrid":
        style_instructions = (
            "Synthesize across the retrieved evidence.\n"
            "If the question asks for both a reported value and an explanation or update, combine the numeric evidence with the narrative explanation in one answer.\n"
            "Prefer using both structured evidence and explanatory text when both are available in the context.\n"
        )
    if answer_style == "multimodal":
        style_instructions = (
            "Synthesize across text, table, and image-derived evidence.\n"
            "For table-related queries, directly extract values from the retrieved table context and perform only simple arithmetic that is explicitly supported by the provided numbers.\n"
            "For chart or figure-related queries, use the image-derived or figure-adjacent context to report the relevant value or trend.\n"
            "When both textual explanation and structured evidence are available, combine them into one concise answer.\n"
            "If the user asks for a comparison of values, present the result as a compact markdown table.\n"
        )

    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are answering questions about the IFC 2024 annual report.\n"
            "Use only the provided context.\n"
            f"{style_instructions}"
            "If the context is incomplete, say so plainly instead of guessing.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )


def build_rag_chain(retriever, answer_style: str = "default"):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        temperature=0,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _prompt_for_style(answer_style)},
    )


def _format_documents(documents) -> str:
    return "\n\n".join(document.page_content for document in documents)


def stream_rag_response(
    retriever,
    query: str,
    *,
    answer_style: str = "default",
    config: dict | None = None,
):
    """Stream a standard RAG answer while preserving source documents for caching/citation."""
    documents = retriever.invoke(query)
    prompt = _prompt_for_style(answer_style).format(
        context=_format_documents(documents),
        question=query,
    )
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        temperature=0,
    )
    for chunk in llm.stream(prompt, config=config):
        yield getattr(chunk, "content", str(chunk)), documents
