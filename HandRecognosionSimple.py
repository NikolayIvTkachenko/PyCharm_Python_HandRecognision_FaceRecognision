import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = 

while True:
    success, img =cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = hands.proccess(imgRGB)

    print(results)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Picture", img)
    cv2.waitKey(1)