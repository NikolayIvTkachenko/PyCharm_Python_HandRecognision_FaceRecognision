import os

import cv2
import numpy as np
import face_recognition
import os


def faceRecognistionVersion02():

    path = 'ImageSampleProject'
    images = []
    classNames = []
    appList = os.listdir(path)
    print(appList)

    for cl in appList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print(classNames)

    encodeListKnow = findEncodings(images)
    print(len(encodeListKnow))
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.5, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
            print(faceDis)



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList




        # faceLoc00 = face_recognition.face_locations(imgFace01)[0]
        # encodeFace00 = face_recognition.face_encodings(imgFace01)[0]
        # print(faceLoc00)
        # cv2.rectangle(imgFace00, (faceLoc00[3], faceLoc00[0]), (faceLoc00[1], faceLoc00[2]), (255, 255, 0), 3)
        #
        # faceLocTest00 = face_recognition.face_locations(imgFaceTest00)[0]
        # encodeFaceTest00 = face_recognition.face_encodings(imgFaceTest00)[0]
        # print(faceLocTest00)
        # cv2.rectangle(imgFaceTest00, (faceLocTest00[3], faceLocTest00[0]), (faceLocTest00[1], faceLocTest00[2]),
        #               (255, 255, 0), 3)
        #
        # results = face_recognition.compare_faces([encodeFace00], encodeFaceTest00)
        # print(results)
        #
        # faceDis = face_recognition.face_distance([encodeFace00], encodeFaceTest00)