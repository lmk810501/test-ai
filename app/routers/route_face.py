import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.config.database.db_conn import db
from app.config.logging.log import myLogger
from app.models.model_face import DeepFaceDto, ReqDeepFaceVo, ResDeepFaceVo
from app.deepface import DeepFace
import tensorflow as tf

router = APIRouter()

tf_version = int(tf.__version__.split('.')[0])

# ------------------------------

if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)


@router.post('/face/regist')
async def face_regist(req: ReqDeepFaceVo, session: Session = Depends(db.session)):
    myLogger.debug('Routes route_face face_regist.py -------------------> start')
    res = ResDeepFaceVo(result_code='S0001', result_message='등록 성공')
    myLogger.debug(req.dict())
    save(req, session)

    global graph

    if tf_version == 1:
        with graph.as_default():
            resp_obj = representWrapper(req)
    elif tf_version == 2:
        resp_obj = representWrapper(req)

    # --------------------------

    myLogger.debug('Routes route_face face_regist.py -------------------> end')

    return res, 200


def save(deepFaceVo: ReqDeepFaceVo, session: Session):
    myLogger.debug('Routes model_face.py -------------------> start')
    deepface = deepFaceVo.dict()
    deepFaceDto = DeepFaceDto()

    embedding = representWrapper(deepFaceVo)
    if embedding == '':
        myLogger.debug('Image is null')
        return deepFaceDto

    deepFaceDto.face_embedding = embedding  # 추후 DeepFace 연동해서 설정 한다
    deepFaceDto.face_id = deepface.get('face_id')
    deepFaceDto.face_nm = deepface.get('face_nm')
    deepFaceDto.face_email = deepface.get('face_email')
    deepFaceDto.face_img = deepface.get('face_img')
    session.add(deepFaceDto)
    session.commit()
    return deepFaceDto


def representWrapper(deepFaceVo):
    model_name = 'VGG-Face'
    distance_metric = 'cosine'
    detector_backend = 'opencv'

    # if 'model_name' in list(req.keys()):
    #     model_name = req['model_name']
    #
    # if 'detector_backend' in list(req.keys()):
    #     detector_backend = req['detector_backend']

    img = deepFaceVo.face_img
    # if 'img' in list(req.keys()):
    #     img = req['img']  # list

    validate_img = False
    if len(img) > 11 and img[0:11] == 'data:image/':
        validate_img = True

    if not validate_img:
        print('invalid image passed!')
        return ''

    # -------------------------------------
    # call represent function from the interface

    try:
        embedding = DeepFace.represent(img
                                       , model_name=model_name
                                       , detector_backend=detector_backend
                                       )
    except Exception as err:
        print('Exception: ', str(err))
        return ''

    return embedding
