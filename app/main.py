import ast
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config.database.db_conn import db, Base
from app.config.env.config import conf
from app.models.model_face import DeepFaceDto, FaceData
from app.routers import route_base, route_face
from app.config.logging.log import myLogger


# def create_app():
#     """
#     앱 함수 실행
#     :return:
#     """
#     app = FastAPI()
#
#     # 데이터 베이스 이니셜라이즈
#     config = conf()
#     conf_dict = asdict(config)
#     db.init_app(app, **conf_dict)
#
#
#     # 레디스 이니셜라이즈
#
#     # 미들웨어 정의
#
#     # 라우터 정의
#     app.include_router(route_base.router)
#     app.include_router(route_face.router)
#
#     return app


def initData():
    database_url = ''.join(['mysql+mysqldb://', conf_dict.get("DB_USER")[0], ':', conf_dict.get("DB_PASSWORD")[0], '@',
                            conf_dict.get("DB_HOST")[0], ':', conf_dict.get("DB_PORT")[0], '/',
                            conf_dict.get("DB_DATABASE"),
                            '?charset-utf8'])
    pool_recycle = conf_dict.setdefault("DB_POOL_RECYCLE", 3000)
    engine = create_engine(
        database_url,
        echo=True,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        connect_args={
            'connect_timeout': 3
        }
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    session = SessionLocal()
    faceData = FaceData()
    if faceData.get_data().empty:
        representations = []
        faceList = session.query(DeepFaceDto).all()
        # pbar = tqdm(range(0, len(faceList)), desc='Finding representations', disable=prog_bar)
        myLogger.debug('init data start ---------------------- > ')
        for face in faceList:
            face_embedding = face.face_embedding
            face_embedding = ast.literal_eval(face_embedding)
            data = [face.face_sn, face.face_id, face.face_nm, face.face_email, face.face_img, face_embedding]
            representations.append(data)
        faceData.load_data(representations)
        myLogger.debug('init data end ---------------------- > ')
    session.close()


app = FastAPI()
config = conf()
conf_dict = asdict(config)
initData()
db.init_app(app, **conf_dict)
app.include_router(route_base.router)
app.include_router(route_face.router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8080, reload=True)
