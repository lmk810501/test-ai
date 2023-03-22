from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy import Column, TEXT, BIGINT, VARCHAR
from typing import Optional
from app.config.database.db_conn import Base
from app.models.model_common import ResCommonVo
from typing import Type
import inspect
import pandas as pd


def as_form(cls: Type[BaseModel]):
    new_parameters = []

    for field_name, model_field in cls.__fields__.items():
        model_field: ModelField  # type: ignore
        new_parameters.append(
            inspect.Parameter(
                model_field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=Form(False) if not model_field.required else Form(...),
                annotation=model_field.outer_type_,
            )
        )

    async def as_form_func(**data):
        return cls(**data)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=new_parameters)
    as_form_func.__signature__ = sig  # type: ignore
    setattr(cls, 'as_form', as_form_func)
    return cls


'''
form data 를 받기 위한 VO
as_form function 필요함
'''


@as_form
class ReqDeepFaceFindVo(BaseModel):
    face_img: UploadFile
    face_nm: str


'''
form data 를 받기 위한 VO
as_form function 필요함
'''


@as_form
class ReqDeepFaceFileVo(BaseModel):
    face_sn: Optional[int]
    face_embedding: Optional[str]
    face_id: str
    face_nm: str
    face_email: str
    face_img: UploadFile


'''
form data 를 받기 위한 VO
as_form function 필요함
'''


@as_form
class ReqDeepFaceDetectVo(BaseModel):
    img_path: str
    img: UploadFile


'''
json body data 를 받기 위한 VO
'''


class ReqDeepFaceVo(BaseModel):
    face_sn: Optional[int]
    face_embedding: Optional[str]
    face_id: str
    face_nm: str
    face_email: str
    face_img: str


## Response
class ResDeepFaceVo(ResCommonVo):
    face_sn: Optional[int]
    face_embedding: Optional[str]
    face_id: Optional[str]
    face_nm: Optional[str]
    face_email: Optional[str]


## Response
class ResDeepFaceFindVo(ResCommonVo):
    face_sn: Optional[int]
    face_id: Optional[str]
    face_nm: Optional[str]
    face_email: Optional[str]
    face_cosine: Optional[str]


class DeepFaceDto(Base):
    __tablename__ = "face"

    face_sn = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    face_embedding = Column(TEXT, nullable=False)
    face_id = Column(VARCHAR, nullable=True)
    face_nm = Column(VARCHAR, nullable=True)
    face_email = Column(VARCHAR, nullable=True)
    face_img = Column(TEXT, nullable=True)


class HistDto(Base):
    __tablename__ = "hist"

    histSn = Column(BIGINT, nullable=False, autoincrement=True, primary_key=True)
    face_sn = Column(BIGINT, nullable=False)
    face_id = Column(VARCHAR, nullable=True)
    face_nm = Column(VARCHAR, nullable=True)
    face_email = Column(VARCHAR, nullable=True)
    req_nm = Column(VARCHAR, nullable=True)


class FaceData:

    _instance = None
    df = pd.DataFrame()

    def __new__(cls) -> 'FaceData':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_data(cls, data) -> None:
        cls.df = pd.DataFrame(data, columns=["face_sn", "face_id", "face_nm", "face_email", "face_img",'face_embedding'])

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        return cls.df














