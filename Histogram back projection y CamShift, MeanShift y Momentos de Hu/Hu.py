import cv2 as cv
import numpy as np
import imutils

img1 = cv.imread('Figura1.jpg')
img2 = cv.imread('Figura2.jpg')
img3 = cv.imread('Figura3.jpg')
img4 = cv.imread('Figura4.jpg')
img5 = cv.imread('Figura5.jpg')

img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
_, img1 = cv.threshold(img1, 0, 255, cv.THRESH_OTSU)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
_, img2 = cv.threshold(img2, 0, 255, cv.THRESH_OTSU)
img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
_, img3 = cv.threshold(img3, 0, 255, cv.THRESH_OTSU)
img4 = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)
_, img4 = cv.threshold(img4, 0, 255, cv.THRESH_OTSU)
img5 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
_, img5 = cv.threshold(img5, 0, 255, cv.THRESH_OTSU)

def CalcMomentos(objeto):
    momentos = cv.moments(objeto)
    Hu = cv.HuMoments(momentos).flatten()
    return Hu

m1 = CalcMomentos(img1)
m2 = CalcMomentos(img2)
m3 = CalcMomentos(img3)
m4 = CalcMomentos(img4)
m5 = CalcMomentos(img5)

print(m5)

captura = cv.VideoCapture(1)
while (captura.isOpened()):
  ret, imagen = captura.read()
  img = np.copy(imagen)
  img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  _, umbral = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
  if ret == True:
    cv.imshow('Original', imagen)
    cv.imshow('Binarizado', umbral)

    c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0

    momentos = cv.moments(umbral)
    momentosHu = cv.HuMoments(momentos).flatten()
    cnts = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
  
# loop over each contour
    for (i, c) in enumerate(cnts):
    # extract the ROI from the image and compute the Hu Moments feature
    # vector for the ROI
        (x, y, w, h) = cv.boundingRect(c)
        roi = umbral[y:y + h, x:x + w]
        moments = cv.HuMoments(cv.moments(roi)).flatten()
        figs = 'f1:'+str(c1)+',f2:'+str(c2)+',f3:'+str(c3)+',f4:'+str(c4)+',f5:'+str(c5)
        cv.putText(img, figs, (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv.imshow('Deteccion', img)

    if cv.waitKey(1) & 0xFF == ord('s'):
        #print('leidos')
        #print(Hu)
        break
    else: break
captura.release()
cv.destroyAllWindows()