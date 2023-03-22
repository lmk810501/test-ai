import ast

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config.database.db_conn import db
from app.models.model_face import FaceData, DeepFaceDto


def initData(session: Session = Depends(db.session)):
    faceData = FaceData()

    if faceData.get_data().empty:
        representations = []
        faceList = session.query(DeepFaceDto).all()
        # pbar = tqdm(range(0, len(faceList)), desc='Finding representations', disable=prog_bar)
        for face in faceList:
            face_embedding = face.face_embedding
            face_embedding = ast.literal_eval(face_embedding)
            data = [face.face_sn, face.face_id, face.face_nm, face.face_email, face.face_img, face_embedding]
            representations.append(data)
        faceData.load_data(representations)