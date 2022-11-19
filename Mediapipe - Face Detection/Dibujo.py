import cv2 as cv
import mediapipe as mp
import math

paint = cv.imread('Fondo.jpg')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            frame = cv.resize(frame, (400, 400), interpolation=cv.INTER_AREA)
            if ret == False:
                break

            height, width, _ = frame.shape
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            cv.imshow('Ventana de Dibujo', paint)
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=5),
                        mp_drawing.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                    longitud = int(math.hypot(x2 - x1 , y2 - y1)/2)
                    cv.circle(frame, (x1, y1), 3,(255,255,0),3)
                    cv.circle(frame, (x2, y2), 3,(255,255,0),3)
                #print(longitud)
                if(longitud < 5):
                    cv.circle(paint, (int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)), longitud, (255, 255, 0), 3)

            cv.imshow('Frame',frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
cap.release()
cv.destroyAllWindows()