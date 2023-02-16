from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.config.logging.log import myLogger
from app.config.database.db_conn import db
from app.models.model_face import DeepFaceDto

router = APIRouter()


@router.get("/")
async def root(session: Session = Depends(db.session)):
    #deepfaces = session.query(DeepFaceDto).all()

    # data = {'nm': 'gggxcvbcgg', 'email': 'sewtt', 'dept': 'kkxcdgkkk'}
    #
    # face = DeepFaceVo(**data)
    # face.face_embedding = 'akshfalsjfhas;hf;askh;kshdf'
    # log.myLogger.debug('face %s', face.dict())
    # dto = DeepFaceDto()
    # dto.face_sn = 10
    # dto.email = '111'
    # dto.face_embedding = '222'
    # dto.nm = '3333'
    # session.query(DeepFaceDto).filter_by(face_sn= 9).update(face.dict())
    # #session.add(dto)
    # session.commit()
    # log.myLogger.debug(face.dict())
    #
    # return session.query(DeepFaceDto).all()
    return 'hello world'


@router.get("/test")
async def root(session: Session = Depends(db.session)):
    face = session.query(DeepFaceDto).all()
    representations = []
    for f in face:
        instance = []
        instance.append(f.face_sn)
        instance.append(f.face_id)
        instance.append(f.face_nm)
        instance.append(f.face_email)
        instance.append(f.face_img)
        representation = f.face_embedding
        instance.append(representation)
        representations.append(instance)
        myLogger.debug(instance)
    return 'test'
