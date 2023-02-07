from fastapi import APIRouter, Depends
import app.config.logging.log as log
from app.config.database.db_conn import db, Base
from sqlalchemy.orm import Session
from sqlalchemy import Column, TEXT, INT, BIGINT, VARCHAR, String
from pydantic import BaseModel

router = APIRouter()


class DeepFace(Base):
    __tablename__ = 'face'
    face_sn = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    face_embedding = Column(TEXT, nullable=True)
    nm = Column(VARCHAR, nullable=True)
    email = Column(VARCHAR, nullable=True)
    dept = Column(VARCHAR, nullable=True)


@router.get("/")
async def root(session: Session = Depends(db.session)):
    deepFaceCreate = DeepFace(face_embedding='11234', nm='11324', email='11gdg', dept='dfgdg11')
    session.add(deepFaceCreate)
    session.commit()

    return 'test'
