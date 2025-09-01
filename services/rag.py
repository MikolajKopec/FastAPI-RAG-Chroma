import hashlib
import tempfile
from pathlib import Path
from db import vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say you don't know.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", "", " "]
)

llm = ChatOllama(model="gpt-oss:20b", temperature=0)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _load_docst_from_tempfile(tmp_path: Path, filename: str):
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(tmp_path).load()
    elif ext == "docx":
        return Docx2txtLoader(tmp_path).load()
    elif ext in ["txt", "md"]:
        return TextLoader(tmp_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _bytes_to_documents(file_bytes: bytes, filename: str) -> list[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = Path(tmp_file.name)
    try:
        docs: list[Document] = _load_docst_from_tempfile(tmp_path, filename)
        for d in docs:
            d.metadata = {**(d.metadata or {}), "filename": filename}
        return docs
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _chunk_documents(docs: list[Document]) -> list[Document]:
    return text_splitter.split_documents(docs)


def _upsert_chunks(
    chunks: list[Document], content_hash: str, metadata: dict = {}
) -> tuple[int, int]:
    texts = [c.page_content for c in chunks]
    metadatas = []
    ids = []
    for i, c in enumerate(chunks):
        md = c.metadata | metadata or metadata
        md["content_hash"] = content_hash
        metadatas.append(md)
        ids.append(f"{content_hash}:{i}")
    added = 0
    try:
        vector_store.vs.add_texts(texts, metadatas=metadatas, ids=ids)
        added = len(texts)
    except Exception as e:
        print(e)
    vector_store.vs.persist()
    return added, len(texts)


def _get_retriver(k: int = 4):
    return vector_store.vs.as_retriever(search_kwargs={"k": k})


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        fn = d.metadata.get("filename", "unknown")
        page = d.metadata.get("page")
        tag = f"{fn}, page {page if page is not None else ''}"
        parts.append(f"[{i}] {tag}\n {d.page_content}")
    return "\n\n".join(parts)


def _build_chain(retriver):
    return (
        {
            "question": RunnablePassthrough(),
            "context": retriver | _format_docs,
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )


def vectorize_file_bytes(file_bytes: bytes, filename: str, metadata: dict = None):
    if not file_bytes:
        raise ValueError("File bytes cannot be empty")
    content_hash = sha256_bytes(file_bytes)
    docs = _bytes_to_documents(file_bytes, filename)
    chunks = _chunk_documents(docs)
    added, total = _upsert_chunks(chunks, content_hash, metadata)
    return {
        "filename": filename,
        "content_hash": content_hash,
        "chunks_total": total,
        "chunks_added": added,
    }


def answer_with_sources(question: str, k: int = 4) -> dict:
    retriver = _get_retriver(k=k)
    docs = retriver.invoke(question)
    chain = _build_chain(retriver)
    answer = chain.invoke(question)
    sources = [
        {
            "filename": d.metadata["filename"],
            "page": d.metadata["page"],
            "content_hash": d.metadata["content_hash"],
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
