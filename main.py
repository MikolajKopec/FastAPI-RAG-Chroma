import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from api.v1.files import files as fr
from api.v1.chat import chat as cr


app = FastAPI()

app.include_router(fr)
app.include_router(cr)


@app.get("/")
async def root() -> dict:
    return {"message": "Hello World"}
