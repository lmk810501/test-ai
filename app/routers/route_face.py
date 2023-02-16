import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request, Form
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from app.config.database.db_conn import db
from app.config.logging.log import myLogger
from app.models.enums.en_exception_type import EnExceptionType
from app.models.model_face import DeepFaceDto, ResDeepFaceVo, ReqDeepFaceFileVo, ReqDeepFaceFindVo, ResDeepFaceFindVo
from app.deepface import CustomDeepFace
import tensorflow as tf
from pathlib import Path

router = APIRouter()


@router.post('/face/find')
async def face_find(req: Request, reqVo: ReqDeepFaceFindVo = Depends(ReqDeepFaceFindVo.as_form),
                    session: Session = Depends(db.session)):
    myLogger.debug('route_face face_find -------------------> start')
    res = ResDeepFaceFindVo()
    enResult = EnExceptionType.S0001
    img = await reqVo.face_img.read()

    # img 확장자 추출
    _, img_ext = os.path.splitext(reqVo.face_img.filename)

    # cv2로 이미지를 읽기 위한 처리
    img_stream = BytesIO(img)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    cv2Img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    df = CustomDeepFace.find(cv2Img, session, enforce_detection=False)

    if len(df) > 0:
        myLogger.debug('route_face face_find df head-------------------> %s', df.head())
        res.face_sn = df.iloc[0, 0]
        res.face_id = df.iloc[0, 1]
        res.face_nm = df.iloc[0, 2]
        res.face_email = df.iloc[0, 3]
        res.face_cosine = str(df.iloc[0, 5])
    else:
        enResult = EnExceptionType.E0002

    res.result_code = enResult.code
    res.result_message = enResult.desc

    result_json = res.dict()
    result_json['face_sn'] = str(result_json['face_sn'])

    return result_json, 200


@router.post('/face/regist')
async def face_regist(req: Request, reqVo: ReqDeepFaceFileVo = Depends(ReqDeepFaceFileVo.as_form),
                      session: Session = Depends(db.session)):
    myLogger.debug('route_Face face_regist -------------------> start')
    enResult = EnExceptionType.S0001
    res = ResDeepFaceVo(result_code=enResult.code, result_message=enResult.desc)
    myLogger.debug('route_Face face_regist------> id = %s, nm = %s, email = %s', reqVo.face_id, reqVo.face_nm,
                   reqVo.face_email)
    try:
        # uploadFile img
        img = await reqVo.face_img.read()
        
        # img 확장자 추출
        _, img_ext = os.path.splitext(reqVo.face_img.filename)
        
        # cv2로 이미지를 읽기 위한 처리
        img_stream = BytesIO(img)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        cv2Img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # img to base64 변경 을 위한 처리
        _, buffer = cv2.imencode(img_ext, cv2Img)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        img_ext = img_ext.replace('.', '')
        base64_prefix = f"data:image/{img_ext};base64,"
        base64_img = base64_prefix + base64_img
        
        # 얼굴 특징점 추출
        embedding = get_embedding(cv2Img)

        # 얼굴 특징 점을 찾지 못하면 오류
        if len(embedding) == 0:
            myLogger.debug('route_Face face_regist embedding is null------------------->')
            enResult = EnExceptionType.E0002
            res.result_code = enResult.code
            res.result_message = enResult.desc
            return res, 200
        
        # DB저장, 얼굴 특징점 및 이미지 base64 string 드읃ㅇ
        save(face_id=reqVo.face_id, face_nm=reqVo.face_nm
             , face_email=reqVo.face_email, face_embedding=embedding
             , face_img=base64_img, session=session)

    except Exception as e:
        myLogger.exception('route_Face face_regist exception-------------------> %s', str(e))
        enResult = EnExceptionType.E0002

    # save(req, session)
    myLogger.debug('route_Face face_regist -------------------> end')

    res = ResDeepFaceVo(result_code=enResult.code, result_message=enResult.desc)

    return res, 200


# 이미지 특징점 추출
def get_embedding(encoding_img):
    model_name = 'VGG-Face'
    detector_backend = 'opencv'
    return CustomDeepFace.represent(encoding_img
                                    , model_name=model_name
                                    , detector_backend=detector_backend
                                    )


# db 저장
def save(face_id: str, face_nm: str, face_email: str, face_img: str, face_embedding: str, session: Session):
    myLogger.debug('Routes model_face.py -------------------> start')
    deepFaceDto = DeepFaceDto()
    deepFaceDto.face_embedding = str(face_embedding)  # 추후 DeepFace 연동해서 설정 한다
    deepFaceDto.face_id = face_id
    deepFaceDto.face_nm = face_nm
    deepFaceDto.face_email = face_email
    deepFaceDto.face_img = face_img
    session.add(deepFaceDto)
    session.commit()
    return deepFaceDto
