from typing import Literal

from langchain_core.tools import tool

from src.llm import get_chat_model


@tool
def select_retrieval_filters(
    content_type: Literal["all", "text", "table", "image"],
    start_page: int = 0,
    end_page: int = 0,
) -> dict:
    """Select retrieval filters for an IFC annual report question."""
    return {
        "content_type": content_type,
        "start_page": max(0, int(start_page)),
        "end_page": max(0, int(end_page)),
    }


def plan_retrieval_filters(query: str) -> dict:
    """Use Gemini function calling to derive optional metadata filters from a query."""
    model = get_chat_model(temperature=0).bind_tools([select_retrieval_filters])
    prompt = (
        "Choose retrieval filters for this IFC annual report question. "
        "Call the tool only when the query clearly asks for a table, chart/image, text-only passage, or page range. "
        "Use 0 for start_page/end_page when no explicit page range is present.\n\n"
        f"Question: {query}"
    )
    result = model.invoke(prompt)
    tool_calls = getattr(result, "tool_calls", None) or []
    if not tool_calls:
        return {"content_type": "all", "start_page": 0, "end_page": 0}

    args = tool_calls[0].get("args", {})
    return select_retrieval_filters.invoke(args)
