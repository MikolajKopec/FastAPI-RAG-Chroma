from fastapi import APIRouter

from services.rag import answer_with_sources


chat = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


@chat.post("/{question}")
def ask_question(question: str) -> dict:
    return answer_with_sources(question)
