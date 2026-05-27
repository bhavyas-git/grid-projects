from langchain_core.callbacks import BaseCallbackHandler

from src.config import LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY


def get_callbacks() -> list[BaseCallbackHandler]:
    if not LANGFUSE_ENABLED or not LANGFUSE_PUBLIC_KEY:
        return []

    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        return []
