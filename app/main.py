from dataclasses import asdict

import uvicorn
from fastapi import FastAPI

from app.config.database.db_conn import db
from app.config.env.config import conf
from app.routers import route_base, route_face

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

app = FastAPI()
config = conf()
conf_dict = asdict(config)
db.init_app(app, **conf_dict)
app.include_router(route_base.router)
app.include_router(route_face.router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8080, reload=True)


