import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
#mp.solutions.hands :This module contains the hand detection model.
hands = mp_hands.Hands()#hand detector object

mp_draw = mp.solutions.drawing_utils 
#visualizes the hand skeleton.

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #Check if Hand Detected
    if results.multi_hand_landmarks:
        for hndlms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hndlms,
                mp_hands.HAND_CONNECTIONS
            )#Draw Hand Skeleton

    cv2.imshow('Hands tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break