import cv2
import numpy as np
import face_recognition


def faceRecognistionVersion01():

    imgFace00 = face_recognition.load_image_file("ImagesSample/richard.jpg")
    imgFace00 = cv2.cvtColor(imgFace00, cv2.COLOR_BGR2RGB)
    imgFaceTest00 = face_recognition.load_image_file("ImagesSample/richardTest.jpg")
    imgFaceTest00 = cv2.cvtColor(imgFaceTest00, cv2.COLOR_BGR2RGB)

    imgFace01 = face_recognition.load_image_file("ImagesSample/Face01.jpg")
    imgFace01 = cv2.cvtColor(imgFace01, cv2.COLOR_BGR2RGB)
    imgFace02 = face_recognition.load_image_file("ImagesSample/Face02.jpg")
    imgFace02 = cv2.cvtColor(imgFace02, cv2.COLOR_BGR2RGB)
    imgFace03 = face_recognition.load_image_file("ImagesSample/Face03.jpg")
    imgFace03 = cv2.cvtColor(imgFace03, cv2.COLOR_BGR2RGB)
    imgFace04 = face_recognition.load_image_file("ImagesSample/Face04.jpg")
    imgFace04 = cv2.cvtColor(imgFace04, cv2.COLOR_BGR2RGB)

    imgPeople01 = face_recognition.load_image_file("ImagesSample/TestPeople01.jpg")
    imgPeople01 = cv2.cvtColor(imgPeople01, cv2.COLOR_BGR2RGB)
    imgPeople02 = face_recognition.load_image_file("ImagesSample/TestPeople02.jpg")
    imgPeople02 = cv2.cvtColor(imgPeople02, cv2.COLOR_BGR2RGB)
    imgPeople03 = face_recognition.load_image_file("ImagesSample/TestPeople03.jpg")
    imgPeople03 = cv2.cvtColor(imgPeople03, cv2.COLOR_BGR2RGB)

    # faceLoc01 = face_recognition.face_locations(imgFace01)[0]
    # encodeFace01 = face_recognition.face_encodings(imgFace01)[0]
    # print(faceLoc01)
    # cv2.rectangle(imgFace01, (faceLoc01[3], faceLoc01[0]), (faceLoc01[1], faceLoc01[2]), (255, 255, 0), 3)

    faceLoc00 = face_recognition.face_locations(imgFace01)[0]
    encodeFace00 = face_recognition.face_encodings(imgFace01)[0]
    print(faceLoc00)
    cv2.rectangle(imgFace00, (faceLoc00[3], faceLoc00[0]), (faceLoc00[1], faceLoc00[2]), (255, 255, 0), 3)

    faceLocTest00 = face_recognition.face_locations(imgFaceTest00)[0]
    encodeFaceTest00 = face_recognition.face_encodings(imgFaceTest00)[0]
    print(faceLocTest00)
    cv2.rectangle(imgFaceTest00, (faceLocTest00[3], faceLocTest00[0]), (faceLocTest00[1], faceLocTest00[2]), (255, 255, 0), 3)

    results = face_recognition.compare_faces([encodeFace00], encodeFaceTest00)
    print(results)

    # peopleLoc01 = face_recognition.face_locations(imgPeople01)[0]
    # encodePeople01 = face_recognition.face_encodings(imgPeople01)[0]
    # print(peopleLoc01)
    # cv2.rectangle(imgPeople01, (peopleLoc01[3], peopleLoc01[0]), (peopleLoc01[1], peopleLoc01[2]), (255, 255, 0), 3)

    #cv2.imshow("Face 01", imgFace01)
    #cv2.imshow("People 01", imgPeople01)

    cv2.imshow("imgFace00", imgFace00)
    cv2.imshow("imgFaceTest00", imgFaceTest00)
    cv2.waitKey(0)
