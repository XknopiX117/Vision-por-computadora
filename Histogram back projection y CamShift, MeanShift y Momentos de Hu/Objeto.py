import cv2 as cv
import numpy as np

img = cv.imread('Objeto.jpeg')
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ref_hist = cv.calcHist([hsv_img], [0], None, [180], [0, 180])

cap = cv.VideoCapture(1)
ret,frame = cap.read()
x, y, w, h = 300, 200, 200, 300
track_window = (x, y, w, h)
roi = frame[y : y + h, x : x + w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
cv.normalize(ref_hist,ref_hist, 0, 255, cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], ref_hist, [0,180], 1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('Deteccion', img2)
        if cv.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break