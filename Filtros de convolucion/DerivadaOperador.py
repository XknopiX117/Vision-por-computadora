import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

def Derivada(c):
    global der
    der = c
    if(Op == 0):
        cv.imshow("Filtro derivada", FiltroSobel(img, der))
    elif(Op == 1):
        if(der == 0):
            kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        else:
            kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        cv.imshow("Filtro derivada", FiltroPrewitt(img, kernel))
    else:
        cv.imshow("Filtro derivada", FiltroScharr(img, der))

def Operador(o):
    global kernel
    global Op
    Op = o
    if(o == 0):
        cv.imshow("Filtro derivada", FiltroSobel(img, der))
    elif(o == 1):
        if(der == 0):
            kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        else:
            kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        cv.imshow("Filtro derivada", FiltroPrewitt(img, kernel))
    else:
        cv.imshow("Filtro derivada", FiltroScharr(img, der))
        


def FiltroSobel(image, d):
    img_der = cv.Sobel(image, -1, int(not d), d)
    return img_der

def FiltroPrewitt(image, matriz):
    img_der = cv.filter2D(image, -1, matriz)
    return img_der

def FiltroScharr(image, d):
    img_der = cv.Scharr(image, -1, int(not d), d)
    return img_der
'''
Imagen 1
'''
img = img1
der = 0
kernel = 0
Op = 0
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Operador", "Filtro derivada", 0, 2, Operador)
cv.createTrackbar("Derivada", "Filtro derivada", 0, 1, Derivada)
cv.imshow("Filtro derivada", FiltroSobel(img, der))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 2
'''
img = img2
der = 0
kernel = 0
Op = 0
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Operador", "Filtro derivada", 0, 2, Operador)
cv.createTrackbar("Derivada", "Filtro derivada", 0, 1, Derivada)
cv.imshow("Filtro derivada", FiltroSobel(img, der))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 3
'''
img = img3
der = 0
kernel = 0
Op = 0
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Operador", "Filtro derivada", 0, 2, Operador)
cv.createTrackbar("Derivada", "Filtro derivada", 0, 1, Derivada)
cv.imshow("Filtro derivada", FiltroSobel(img, der))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 4
'''
img = img4
der = 0
kernel = 0
Op = 0
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Operador", "Filtro derivada", 0, 2, Operador)
cv.createTrackbar("Derivada", "Filtro derivada", 0, 1, Derivada)
cv.imshow("Filtro derivada", FiltroSobel(img, der))
cv.waitKey(0)
cv.destroyAllWindows()