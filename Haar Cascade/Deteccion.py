import cv2 as cv
import numpy as np

CarnageClassif = cv.CascadeClassifier('cascade.xml')

cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    objeto = CarnageClassif.detectMultiScale(gray, scaleFactor = 8, minNeighbors = 100)

    for (w, h, x, y) in objeto:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv.putText(frame, 'Carnage', (x,y), 2, 0.7, (0,255,0), 2, cv.LINE_AA)

    cv.imshow('Deteccion', frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()