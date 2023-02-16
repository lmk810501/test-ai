import ast
import warnings

from app.models.model_face import DeepFaceDto

warnings.filterwarnings("ignore")

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from app.deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, \
    SFace, Boosting
from app.deepface.commons import functions, distance as dst
from app.deepface.extendedmodels import Age, Gender, Race, Emotion

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)


def build_model(model_name):
    global model_obj

    models = {
        'VGG-Face': VGGFace.loadModel,
        'OpenFace': OpenFace.loadModel,
        'Facenet': Facenet.loadModel,
        'Facenet512': Facenet512.loadModel,
        'DeepFace': FbDeepFace.loadModel,
        'DeepID': DeepID.loadModel,
        'Dlib': DlibWrapper.loadModel,
        'ArcFace': ArcFace.loadModel,
        'SFace': SFace.load_model,
        'Emotion': Emotion.loadModel,
        'Age': Age.loadModel,
        'Gender': Gender.loadModel,
        'Race': Race.loadModel
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        # print(model_name," built")
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]


def find(findImg=None, session=None, model_name='VGG-Face', distance_metric='cosine', model=None,
         enforce_detection=True, detector_backend='opencv', align=True, prog_bar=True, normalization='base',
         silent=False):
    # 모델 설정 추후 여러가지 테스트 해 봐야함
    # Ensemble 관련 테스트 필요
    if model == None:

        if model_name == 'Ensemble':
            if not silent: print("Ensemble learning enabled")
            models = Boosting.loadModel()

        else:  # model is not ensemble
            model = build_model(model_name)
            models = {}
            models[model_name] = model

    else:  # model != None
        if not silent: print("Already built model is passed")

        if model_name == 'Ensemble':
            Boosting.validate_model(model)
            models = model.copy()
        else:
            models = {}
            models[model_name] = model

    # ---------------------------------------

    if model_name == 'Ensemble':
        model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
        metric_names = ['cosine', 'euclidean', 'euclidean_l2']
    elif model_name != 'Ensemble':
        model_names = []
        metric_names = []
        model_names.append(model_name)
        metric_names.append(distance_metric)

    # ---------------------------------------

    # 기존에 디렉토리 읽어서 사용하는 부분 DB 사용 하는 걸로 변경
    representations = []

    faceList = session.query(DeepFaceDto).all()

    pbar = tqdm(range(0, len(faceList)), desc='Finding representations', disable=prog_bar)

    for face in faceList:
        data = [face.face_sn, face.face_id, face.face_nm, face.face_email, face.face_img, face.face_embedding]
        representations.append(data)

    if model_name != 'Ensemble':
        embeddingName = '{}_representation'.format(model_name)
        df = pd.DataFrame(representations, columns=["face_sn", "face_id", "face_nm", "face_email", "face_img", embeddingName])
    else:  # ensemble learning
        columns = ['identity']
        [columns.append('%s_representation' % i) for i in model_names]
        df = pd.DataFrame(representations, columns=columns)

    df_base = df.copy()  # df will be filtered in each img. we will restore it for the next item.
    resp_obj = []

    for j in model_names:

        custom_model = models[j]

        # -------------------------------------
        # call represent function from the interface

        try:
            embedding = represent(findImg
                                  , model_name=model_name
                                  , detector_backend=detector_backend
                                  )
        except Exception as err:
            print('Exception: ', str(err))
            return ''

        for k in metric_names:
            distances = []
            for index, instance in df.iterrows():
                source_representation = instance["%s_representation" % (j)]
                source_representation = ast.literal_eval(source_representation)


                if k == 'cosine':
                    distance = dst.findCosineDistance(source_representation, embedding)
                elif k == 'euclidean':
                    distance = dst.findEuclideanDistance(source_representation, embedding)
                elif k == 'euclidean_l2':
                    distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation),
                                                         dst.l2_normalize(embedding))

                distances.append(distance)

            # ---------------------------

            if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
                continue
            else:
                df["%s_%s" % (j, k)] = distances

                if model_name != 'Ensemble':
                    threshold = dst.findThreshold(j, k)
                    df = df.drop(columns=["%s_representation" % (j)])
                    df = df[df["%s_%s" % (j, k)] <= threshold]

                    df = df.sort_values(by=["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)

                    resp_obj.append(df)
                    df = df_base.copy()  # restore df for the next iteration

    # ----------------------------------

    if model_name == 'Ensemble':

        feature_names = []
        for j in model_names:
            for k in metric_names:
                if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
                    continue
                else:
                    feature = '%s_%s' % (j, k)
                    feature_names.append(feature)

        # print(df.head())

        x = df[feature_names].values

        # --------------------------------------

        boosted_tree = Boosting.build_gbm()

        y = boosted_tree.predict(x)

        verified_labels = [];
        scores = []
        for i in y:
            verified = np.argmax(i) == 1
            score = i[np.argmax(i)]

            verified_labels.append(verified)
            scores.append(score)

        df['verified'] = verified_labels
        df['score'] = scores

        df = df[df.verified == True]
        # df = df[df.score > 0.99] #confidence score
        df = df.sort_values(by=["score"], ascending=False).reset_index(drop=True)
        df = df[['identity', 'verified', 'score']]

        resp_obj.append(df)
        df = df_base.copy()  # restore df for the next iteration

    if len(resp_obj) == 1:
        return resp_obj[0]

    return resp_obj


def represent(img_path, model_name='VGG-Face', model=None, enforce_detection=True, detector_backend='opencv',
              align=True, normalization='base'):
    if model is None:
        model = build_model(model_name)

    input_shape_x, input_shape_y = functions.find_input_shape(model)

    img = functions.preprocess_face(img=img_path
                                    , target_size=(input_shape_y, input_shape_x)
                                    , enforce_detection=enforce_detection
                                    , detector_backend=detector_backend
                                    , align=align)

    img = functions.normalize_input(img=img, normalization=normalization)

    if "keras" in str(type(model)):
        embedding = model.predict(img, verbose=0)[0].tolist()
    else:
        embedding = model.predict(img)[0].tolist()

    return embedding