import base64
import os
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request, Form
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from app.config.database.db_conn import db
from app.config.logging.log import myLogger, detectedSuccessLogger, detectedFailLogger
from app.models.enums.en_exception_type import EnExceptionType
from app.models.model_face import DeepFaceDto, ResDeepFaceVo, ReqDeepFaceFileVo, ReqDeepFaceFindVo, ResDeepFaceFindVo, \
    ReqDeepFaceDetectVo, HistDto
from app.deepface import CustomDeepFace

import matplotlib.pyplot as plt
import matplotlib.image as pltImg


router = APIRouter()

# 전체 : , hist : 1284, 성공 1042, 실패 :242
# detector_backend = 'opencv'
# model_name = 'Facenet512'
# distance_metric = 'euclidean_l2'
# normalization = 'Facenet512'

# 전체 : 1295, hist : 1029, 성공 :  1010, 실패 : 19, threshold : 10
# 전체 : 1295, hist : 1267, 성공 :  1176, 실패 : 91, threshold : 12
# detector_backend = 'mtcnn'
# model_name = 'Facenet'
# distance_metric = 'euclidean'
# normalization = 'Facenet'

# 전체 : 1282, hist : 1026, 성공 :  924, 실패 : 102
# detector_backend = 'opencv'
# model_name = 'Facenet'
# distance_metric = 'euclidean'
# normalization = 'Facenet'

# 전체 : 1294, hist : 1075, 성공 :  1059, 실패 : 16, threshold : 10
# 전체 : 1294, hist : 1263, 성공 :  1214, 실패 : 49, threshold : 12
# 전체 : 1677, hist : 1632, 성공 :  1494, 실패 : 138, threshold : 12
detector_backend = 'dlib'
model_name = 'Facenet'
distance_metric = 'euclidean'
normalization = 'Facenet'


@router.post('/face/find')
async def face_find(req: Request, reqVo: ReqDeepFaceFindVo = Depends(ReqDeepFaceFindVo.as_form),
                    session: Session = Depends(db.session)):
    myLogger.debug('route_face face_find -------------------> start')
    res = ResDeepFaceFindVo()
    try:
        enResult = EnExceptionType.S0001
        img = await reqVo.face_img.read()

        # img 확장자 추출
        _, img_ext = os.path.splitext(reqVo.face_img.filename)

        # cv2로 이미지를 읽기 위한 처리
        cv2Img = get_cv2Img(img)
        df = CustomDeepFace.find(cv2Img, session, enforce_detection=False, model_name=model_name,
                                 detector_backend=detector_backend, distance_metric=distance_metric, normalization=normalization)

        if len(df) > 0:
            myLogger.debug('route_face face_find df head-------------------> %s', df.head())
            res.face_sn = df.iloc[0, 0]
            res.face_id = df.iloc[0, 1]
            res.face_nm = df.iloc[0, 2]
            res.face_email = df.iloc[0, 3]
            res.face_cosine = str(df.iloc[0, 5])
            myLogger.debug(df)
        else:
            enResult = EnExceptionType.E0002
            detectedFailLogger.debug('route_Face face_find fails df is empty -------------------> %s', reqVo.face_nm)

        res.result_code = enResult.code
        res.result_message = enResult.desc

        result_json = res.dict()
        result_json['face_sn'] = str(result_json['face_sn'])

    except Exception as e:
        enResult = EnExceptionType.E0002
        myLogger.debug('route_face face_find exception -------------------> %s', str(e))
        res.result_code = enResult.code
        res.result_message = enResult.desc
        result_json = res.dict()

    if EnExceptionType.S0001 == enResult:
        detectedSuccessLogger.debug('route_Face face_find success -------------------> %s', reqVo.face_nm)
        saveHist(res.face_sn, res.face_id, res.face_nm, res.face_email, reqVo.face_nm, session)
    else:
        detectedFailLogger.debug('route_Face face_find fails -------------------> %s', reqVo.face_nm)

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
        cv2Img = get_cv2Img(img)

        # img to base64 변경 을 위한 처리
        base64_img = get_base64Img(img_ext, cv2Img)

        # 얼굴 특징점 추출
        embedding = get_embedding(cv2Img)

        # 얼굴 특징 점을 찾지 못하면 오류
        if len(embedding) == 0:
            myLogger.debug('route_Face face_regist embedding is null------------------->')
            enResult = EnExceptionType.E0002
            res.result_code = enResult.code
            res.result_message = enResult.desc
            detectedFailLogger.debug('route_Face face_regist fail embedding is null -------------------> %s',
                                     reqVo.face_nm)
            return res, 200

        # DB저장, 얼굴 특징점 및 이미지 base64 string 드읃ㅇ
        saveFace(face_id=reqVo.face_id, face_nm=reqVo.face_nm
                 , face_email=reqVo.face_email, face_embedding=embedding
                 , face_img=base64_img, session=session)

        detectedSuccessLogger.debug('route_Face face_detect success -------------------> %s', reqVo.face_nm)

    except Exception as e:
        myLogger.exception('route_Face face_regist exception-------------------> %s', str(e))
        detectedFailLogger.debug('route_Face face_regist fail -------------------> %s', reqVo.face_nm)
        enResult = EnExceptionType.E0002

    myLogger.debug('route_Face face_regist -------------------> end')

    res = ResDeepFaceVo(result_code=enResult.code, result_message=enResult.desc)

    return res, 200


@router.post('/face/detect')
async def face_detect(req: Request, reqVo: ReqDeepFaceDetectVo = Depends(ReqDeepFaceDetectVo.as_form),
                      session: Session = Depends(db.session)):
    myLogger.debug('route_Face face_detect -------------------> start')
    enResult = EnExceptionType.S0001
    res = ResDeepFaceVo(result_code=enResult.code, result_message=enResult.desc)
    try:
        # uploadFile img
        img = await reqVo.img.read()

        # img 확장자 추출
        _, img_ext = os.path.splitext(reqVo.img.filename)
        # cv2로 이미지를 읽기 위한 처리
        cv2Img = get_cv2Img(img)
        # img to base64 변경 을 위한 처리
        base64_img = get_base64Img(img_ext, cv2Img)
        dectect_img = CustomDeepFace.detected_face(base64_img, model_name=model_name, model=None,
                                                   enforce_detection=True, detector_backend=detector_backend,
                                                   align=True, normalization=normalization)

        detectedSuccessLogger.debug('route_Face face_detect success -------------------> %s', reqVo.img_path)

    except Exception as e:
        myLogger.exception('route_Face face_detect exception-------------------> %s', str(e))
        detectedFailLogger.debug('route_Face face_detect fail -------------------> %s', reqVo.img_path)
        enResult = EnExceptionType.E0002

    # save(req, session)

    res = ResDeepFaceVo(result_code=enResult.code, result_message=enResult.desc)

    return res, 200


# cv2 Img 읽기
def get_cv2Img(img):
    # cv2로 이미지를 읽기 위한 처리
    #img_stream = BytesIO(img)
    #img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    cv2Img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    return cv2Img


# base64 Encoding
def get_base64Img(img_ext, cv2Img):
    _, buffer = cv2.imencode(img_ext, cv2Img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    img_ext = img_ext.replace('.', '')
    base64_prefix = f"data:image/{img_ext};base64,"
    base64_img = base64_prefix + base64_img
    return base64_img


# 이미지 특징점 추출
def get_embedding(encoding_img):
    return CustomDeepFace.represent(img_path=encoding_img
                                    , model_name=model_name
                                    , enforce_detection=True
                                    , detector_backend=detector_backend
                                    , align=True
                                    , normalization=normalization
                                    )


# db 저장
def saveFace(face_id: str, face_nm: str, face_email: str, face_img: str, face_embedding: str, session: Session):
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


def saveHist(face_sn: int, face_id: str, face_nm: str, face_email: str, req_nm: str, session: Session):
    myLogger.debug('Routes model_face.py -------------------> start')
    histDto = HistDto()
    histDto.face_sn = face_sn  # 추후 DeepFace 연동해서 설정 한다
    histDto.face_id = face_id
    histDto.face_nm = face_nm
    histDto.face_email = face_email
    histDto.req_nm = req_nm
    session.add(histDto)
    session.commit()
    return histDto
