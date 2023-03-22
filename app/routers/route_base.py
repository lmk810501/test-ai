import base64

from fastapi import APIRouter, Depends
from matplotlib import pyplot as plt
from sqlalchemy.orm import Session
from app.config.logging.log import myLogger, detectedSuccessLogger, detectedFailLogger
from app.config.database.db_conn import db
from app.models.model_face import DeepFaceDto
import os
import json
import requests
import aiohttp
import asyncio
import time
from app.models.model_face import DeepFaceDto, ResDeepFaceVo, ReqDeepFaceFileVo, ReqDeepFaceFindVo, ResDeepFaceFindVo, \
    ReqDeepFaceDetectVo, HistDto
import cv2
import numpy as np
from io import BytesIO
from app.deepface import DeepFace, CustomDeepFace

router = APIRouter()
# search_dir = 'E:\\99.lmk810501\\13.AI\\deepface-master\\tests\\success_fail_dataset'
#search_dir = 'E:\\99.lmk810501\\13.AI\\lfw-data'
#search_dir = 'E:\\99.lmk810501\\13.AI\\lfw-data-1295\\lfw-data-1295'
search_dir = 'E:\\99.lmk810501\\13.AI\\lfw-data-2'
# url = 'http://127.0.0.1:8080/face/detect'
#url = 'http://127.0.0.1:8080/face/regist'
url = 'http://127.0.0.1:8080/face/find'


async def post_detect_async(file, data):
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field('img', file)
        for key, value in data.items():
            form.add_field(key, value)
        async with session.post(url, data=form) as resp:
            response_text = await resp.text()
            return response_text


async def post_find_async(file, data):
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field('face_img', file)
        for key, value in data.items():
            form.add_field(key, value)
        async with session.post(url, data=form) as resp:
            response_text = await resp.text()
            return response_text


async def post_async(file, data):
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field('face_img', file)
        for key, value in data.items():
            form.add_field(key, value)
        async with session.post(url, data=form) as resp:
            response_text = await resp.text()
            return response_text


async def resit_file(directory, extension):
    myLogger.debug('search_files start --------------->')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                dir_path = os.path.dirname(file_path)
                dir_name = os.path.basename(dir_path)
                file_name = os.path.basename(file_path)
                #if '_r.jpg' in file_path:
                if '_0001.jpg' in file_path:
                    face_id = 'test_id_' + dir_name
                    face_nm = file_name
                    email = dir_name + '@email.com'
                    with open(file_path, 'rb') as face_img:
                        data = {
                            'face_id': face_id,
                            'face_nm': face_nm,
                            'face_email': email
                        }
                        myLogger.debug('search_files data --------------->%s, %s', file_path, json.dumps(data))
                        response = await post_async(face_img, data)
                        myLogger.debug('search_files response --------------->%s', response)

    # myLogger.debug('search_files start --------------->')
    # for root, dirs, files in os.walk(directory):
    #     #myLogger.debug('root---------------->' + root)
    #     isRv = False
    #     for file in files:
    #         if file.endswith(extension):
    #             file_path = os.path.join(root, file)
    #             dir_path = os.path.dirname(file_path)
    #             dir_name = os.path.basename(dir_path)
    #             file_name = os.path.basename(file_path)
    #             if '_v.jpg' in file_path:
    #                 myLogger.debug('file_name---------------->' + file_name)
    #                 face_id = 'test_id_' + dir_name
    #                 face_nm = file_name
    #                 email = dir_name + '@email.com'
    #                 isRv = True
    #                 with open(file_path, 'rb') as face_img:
    #                     data = {
    #                         'face_id': face_id,
    #                         'face_nm': face_nm,
    #                         'face_email': email
    #                     }
    #                     # myLogger.debug('search_files data --------------->%s, %s', file_path, json.dumps(data))
    #                     # response = await post_async(face_img, data)
    #                     # myLogger.debug('search_files response --------------->%s', response)
    #     if isRv:
    #         detectedSuccessLogger.debug(root)
    #     else:
    #         detectedFailLogger.debug(root)


async def detect_file(directory, extension):
    myLogger.debug('detect_file start --------------->')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                dir_path = os.path.dirname(file_path)
                if '0001' in file_path:
                    with open(file_path, 'rb') as face_img:
                        data = {
                            'img_path': file_path,
                        }
                        myLogger.debug('detect_file data --------------->%s, %s', file_path, json.dumps(data))
                        response = await post_detect_async(face_img, data)
                        myLogger.debug('detect_file response --------------->%s', response)


async def find_file(directory, extension):
    myLogger.debug('find_file start --------------->')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                # dir_path = os.path.dirname(file_path)
                face_nm = os.path.basename(file_path)
                myLogger.debug('facenm--->' + face_nm)
                #if '_v.jpg' in file_path:
                if '_0002.jpg' in file_path:
                    if 'Binyamin_Ben-Eliezer' not in file_path or 'Elisabeth_Schumacher' not in file_path or 'Muammar_Gaddafi' not in file_path:
                        with open(file_path, 'rb') as face_img:
                            data = {
                                'face_nm': face_nm,
                            }
                            myLogger.debug('find_file data --------------->%s, %s', file_path, json.dumps(data))
                            time.sleep(1)
                            response = await post_find_async(face_img, data)
                            myLogger.debug('find_file response --------------->%s', response)



async def compare(directory, extension):
    detector_backend = 'dlib'
    model_name = 'VGG-Face'
    distance_metric = 'cosine'
    normalization = 'raw'

    rPath = 'E:\\99.lmk810501\\13.AI\\lfw-data-1295\\img_r\\'
    vPath = 'E:\\99.lmk810501\\13.AI\\lfw-data-1295\\img_v\\'

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                # dir_path = os.path.dirname(file_path)
                face_nm = os.path.basename(file_path)
                #if '_r.jpg' in file_path or '_v.jpg' in file_path:
                if 'Aaron_Peirsol_r.jpg' in file_path:
                    with open(file_path, 'rb') as face_img:

                        cv2Img = get_cv2Img(face_img.read())
                        # img to base64 변경 을 위한 처리
                        base64_img = get_base64Img('.jpg', cv2Img)

                        dectect_img = CustomDeepFace.detected_face(base64_img, model_name=model_name, model=None,
                                                                   enforce_detection=False,
                                                                   detector_backend=detector_backend,
                                                                   align=True, normalization=normalization)

                        changeImg = np.squeeze(dectect_img, axis=0)
                        if '_r.jpg' in file_path:
                            saveFilePath = os.path.join(rPath, face_nm)
                        else:
                            saveFilePath = os.path.join(vPath, face_nm)

                        cv2.imwrite(saveFilePath, changeImg)




@router.post("/find")
async def root(reqVo: ReqDeepFaceFindVo = Depends(ReqDeepFaceFindVo.as_form),
               session: Session = Depends(db.session)):
    return 'hello world'


@router.get("/lfw/regist")
async def root(session: Session = Depends(db.session)):
    await resit_file(search_dir, 'jpg')
    return 'test'


@router.get("/lfw/detect")
async def root(session: Session = Depends(db.session)):
    await detect_file(search_dir, 'jpg')
    return 'test'


@router.get("/lfw/find")
async def root(session: Session = Depends(db.session)):
    await find_file(search_dir, 'jpg')
    return 'test'


@router.get("/lfw/compare")
async def root(session: Session = Depends(db.session)):
    await compare(search_dir, 'jpg')
    return 'test'


# cv2 Img 읽기
def get_cv2Img(img):
    # cv2로 이미지를 읽기 위한 처리
    # img_stream = BytesIO(img)
    # img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
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
