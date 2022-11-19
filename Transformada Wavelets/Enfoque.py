import cv2 as cv
import numpy as np
import pywt as pw

img1 = cv.imread('MED2B.JPG', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('MED2A.JPG', cv.IMREAD_GRAYSCALE)

img1 = cv.resize(img1, (0, 0), fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
img2 = cv.resize(img2, (0, 0), fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)

def Metodo(coeffs1, coeffs2, metodo):
    if(metodo == 'prom'):
        coeffs = (coeffs1 + coeffs2) / 2
    elif(metodo == 'min'):
        coeffs = np.minimum(coeffs1, coeffs2)
    elif(metodo == 'max'):
        coeffs = np.maximum(coeffs1, coeffs2)
    return coeffs

def Enfocar(image1, image2, metodo, wavelet_name):
    union = []
    wavelet = pw.Wavelet(wavelet_name)
    coeffs1 = pw.wavedec2(image1[:,:], wavelet)
    coeffs2 = pw.wavedec2(image2[:,:], wavelet)

    for i in range(len(coeffs1) - 1):
        if (i == 0):
            union.append(Metodo(coeffs1[0], coeffs2[0], metodo))
        else:
            c0 = Metodo(coeffs1[i][0], coeffs2[i][0], metodo)
            c1 = Metodo(coeffs1[i][1], coeffs2[i][1], metodo)
            c2 = Metodo(coeffs1[i][2], coeffs2[i][2], metodo)
            union.append((c0, c1, c2))
    
    new_img = pw.waverec2(union, wavelet)
    new_img = np.multiply(np.divide(new_img - np.min(new_img),(np.max(new_img) - np.min(new_img))),255)
    new_img = new_img.astype(np.uint8)
    return new_img

#Wavelet Haar
img_union = Enfocar(img1, img2, 'prom', 'haar')
cv.imshow('Enfoque promedio Haar', img_union)
img_union = Enfocar(img1, img2, 'min', 'haar')
cv.imshow('Enfoque minimo Haar', img_union)
img_union = Enfocar(img1, img2, 'max', 'haar')
cv.imshow('Enfoque maximo Haar', img_union)
cv.waitKey(0)
cv.destroyAllWindows()

#Wavelet Daub2
img_union = Enfocar(img1, img2, 'prom', 'db2')
cv.imshow('Enfoque promedio Daub2', img_union)
img_union = Enfocar(img1, img2, 'min', 'db2')
cv.imshow('Enfoque minimo Daub2', img_union)
img_union = Enfocar(img1, img2, 'max', 'db2')
cv.imshow('Enfoque maximo Daub2', img_union)
cv.waitKey(0)
cv.destroyAllWindows()

#Wavelet Daub4
img_union = Enfocar(img1, img2, 'prom', 'db4')
cv.imshow('Enfoque promedio Daub4', img_union)
img_union = Enfocar(img1, img2, 'min', 'db4')
cv.imshow('Enfoque minimo Daub4', img_union)
img_union = Enfocar(img1, img2, 'max', 'db4')
cv.imshow('Enfoque maximo Daub4', img_union)
cv.waitKey(0)
cv.destroyAllWindows()