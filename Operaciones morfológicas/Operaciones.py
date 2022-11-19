import cv2 as cv
import numpy as np

img = cv.imread('ryuhayabusa.jpg')

#Blanco y negro
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

#Dilatacion
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
img_dil = cv.dilate(img, kernel, -1)
cv.imshow('Original', img)
cv.imshow('Dilatacion', img_dil)
cv.waitKey(0)
cv.destroyAllWindows()

#Erosion
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
img_ero = cv.erode(img, kernel, -1)
cv.imshow('Original', img)
cv.imshow('Erosion', img_ero)
cv.waitKey(0)
cv.destroyAllWindows()

#Gradiente
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_grad = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('Original', img)
cv.imshow('Gradiente', img_grad)
cv.waitKey(0)
cv.destroyAllWindows()

#Apertura
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_op = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv.imshow('Original', img)
cv.imshow('Apertura', img_op)
cv.waitKey(0)
cv.destroyAllWindows()

#Cierre
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_cs = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('Original', img)
cv.imshow('Cierre', img_cs)
cv.waitKey(0)
cv.destroyAllWindows()

#Hit or Miss
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
img_hm = cv.morphologyEx(img, cv.MORPH_HITMISS, kernel)
cv.imshow('Original', img)
cv.imshow('Hit or Miss', img_hm)
cv.waitKey(0)
cv.destroyAllWindows()

#Extraccion de bordes
kernel = cv.getStructuringElement(cv.MORPH_ERODE, (3,3))
img_hm = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('Original', img)
cv.imshow('Extraccion de bordes', img_hm)
cv.waitKey(0)
cv.destroyAllWindows()

#Esqueleto
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
skeleton = np.zeros(img.shape, np.uint8)
eroded = np.zeros(img.shape, np.uint8)
temp = np.zeros(img.shape, np.uint8)
thresh = img.copy()
img_ex = 0
flag = True
while(flag):
    cv.erode(thresh, kernel, eroded)
    cv.dilate(eroded, kernel, temp)
    cv.subtract(thresh, temp, temp)
    cv.bitwise_or(skeleton, temp, skeleton)
    thresh, eroded = eroded, thresh
    if cv.countNonZero(thresh) == 0:
        img_ex = skeleton
        flag = False
cv.imshow('Original', img)
cv.imshow('Esqueleto', img_ex)
cv.waitKey(0)
cv.destroyAllWindows()

#TopHat
img_aux = cv.imread('ryuhayabusa.jpg', 0)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_th = cv.morphologyEx(img_aux, cv.MORPH_TOPHAT, kernel)
cv.imshow('Original', img_aux)
cv.imshow('Top Hat', img_th)
cv.waitKey(0)
cv.destroyAllWindows()

#BottomHat
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_op_aux = cv.morphologyEx(img_aux, cv.MORPH_OPEN, kernel)
img_bh = img_aux - img_op_aux
cv.imshow('Original', img_aux)
cv.imshow('Bottom Hat', img_bh)
cv.waitKey(0)
cv.destroyAllWindows()

#BlackHat
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_op_aux = cv.morphologyEx(img_aux, cv.MORPH_BLACKHAT, kernel)
img_blh = img_aux - img_op_aux
cv.imshow('Original', img_aux)
cv.imshow('Black Hat', img_blh)
cv.waitKey(0)
cv.destroyAllWindows()

#Gradiente moforlogico
img_aux1, img_aux2 = img_aux, img_aux
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
img_dil_aux = cv.morphologyEx(img_aux1, cv.MORPH_DILATE, kernel)
img_ero_aux = cv.morphologyEx(img_aux2, cv.MORPH_ERODE, kernel)
img_grad_morf = img_dil_aux - img_ero_aux
cv.imshow('Original', img_aux)
cv.imshow('Gradiente morfologico', img_grad_morf)
cv.waitKey(0)
cv.destroyAllWindows()

#Relleno de regiones
im_fill = img.copy()
h, w = img.shape
mask = np.zeros((h + 2, w + 2), np.uint8)
im_flood_fill = im_fill.astype("uint8")
cv.floodFill(im_flood_fill, mask, (0, 0), 255)
im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
img_ff = img | im_flood_fill_inv
cv.imshow('Original', img)
cv.imshow('Relleno de regiones', img_ff)
cv.waitKey(0)
cv.destroyAllWindows()

#Suavizacion morfologica
img_aux1 = img_aux
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
img_op_aux = cv.morphologyEx(img_aux1, cv.MORPH_OPEN, kernel)
img_cr_aux = cv.morphologyEx(img_op_aux, cv.MORPH_CLOSE, kernel)
cv.imshow('Original', img_aux)
cv.imshow('Suavizacion morfologica', img_cr_aux)
cv.waitKey(0)
cv.destroyAllWindows()

#Adelgazamiento
thinned = cv.ximgproc.thinning(img)
cv.imshow('Original', img)
cv.imshow('Adelgazamiento', thinned)
cv.waitKey(0)
cv.destroyAllWindows()