import datetime
from pydantic import BaseModel


class FileRecord(BaseModel):
    id: int
    filename: str
    content_type: str
    upload_time: datetime.datetime
    size: int
    file_data: bytes


class FileResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    upload_time: datetime.datetime
    size: int
