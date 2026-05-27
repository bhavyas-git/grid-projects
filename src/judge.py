from pydantic import BaseModel, Field

from src.llm import get_chat_model


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall answer quality score from 1 to 10.")
    rationale: str = Field(description="Brief justification focused on correctness and completeness.")


def judge_answer(question: str, generated_answer: str, ground_truth: str) -> dict:
    model = get_chat_model(temperature=0).with_structured_output(JudgeScore)
    prompt = (
        "You are grading a RAG answer against a reference answer.\n"
        "Score the generated answer from 1 to 10.\n"
        "Prioritize factual correctness, completeness, and whether the answer directly resolves the question.\n"
        "Keep the rationale brief.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {ground_truth}\n"
        f"Generated answer: {generated_answer}\n"
    )
    result = model.invoke(prompt)
    return result.model_dump()
