from fastapi import APIRouter, Depends

from services.rag import RAGHandler, get_rag_handler


chat = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


@chat.post("/{question}")
def ask_question(
    question: str, rag_handler: RAGHandler = Depends(get_rag_handler)
) -> dict:
    return rag_handler.answer_with_sources(question)
