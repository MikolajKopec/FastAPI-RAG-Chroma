import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from db import VectorStore, get_vector_store
from services.rag import vectorize_file_bytes

files = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={404: {"description": "Not found"}},
)


@files.get("/")
async def get_files(store: VectorStore = Depends(get_vector_store)):
    x = store.get_all()
    print(x)
    return x


@files.post("/")
async def upload_file(file: UploadFile = File(...)) -> dict:
    allowed_extensions: list[str] = (".pdf", ".doc", ".docx")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file extension.")
    allowed_content_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    file_content = await file.read()
    return vectorize_file_bytes(
        file_bytes=file_content,
        filename=file.filename,
        metadata={
            "filename": file.filename,
            "uploaded_at": datetime.datetime.now().isoformat(),
        },
    )
