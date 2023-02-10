from pydantic import BaseModel
from sqlalchemy import Column, TEXT, BIGINT, VARCHAR
from typing import Optional
from app.config.database.db_conn import Base
from app.models.model_common import ResCommonVo


class ReqDeepFaceVo(BaseModel):
    face_sn: Optional[int]
    face_embedding: Optional[str]
    face_id: str
    face_nm: str
    face_email: str
    face_img: str


class ResDeepFaceVo(ResCommonVo):
    face_sn: Optional[int]
    face_embedding: Optional[str]
    face_id: Optional[str]
    face_nm: Optional[str]
    face_email: Optional[str]


class DeepFaceDto(Base):
    __tablename__ = "face"

    face_sn = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    face_embedding = Column(TEXT, nullable=False)
    face_id = Column(VARCHAR, nullable=True)
    face_nm = Column(VARCHAR, nullable=True)
    face_email = Column(VARCHAR, nullable=True)
    face_img = Column(TEXT, nullable=True)
