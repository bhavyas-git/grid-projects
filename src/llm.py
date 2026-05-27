import os

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL


def get_genai_client() -> genai.Client | None:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    if not project:
        return None
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


def get_chat_model(temperature: float = 0) -> ChatGoogleGenerativeAI:
    get_genai_client()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        temperature=temperature,
    )
