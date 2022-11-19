import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

def CambioVecindad(c):
    global kernel
    if(c == 0):
        kernel = np.array([[-1, 1]])
        cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
    elif(c == 1):
        kernel = np.array([[-1], [1]])
        cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
    elif(c == 2):
        kernel = np.array([[-1, 0],[0, 1]])
        cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
    else:
        kernel = np.array([[0, -1],[1, 0]])
        cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))

def FiltroDerivada(image, value):
    img_der = cv.filter2D(image, -1, value)
    return img_der

'''
Imagen 1
'''
kernel = np.array([[-1], [1]])
img = img1
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Derivada", "Filtro derivada", 1, 3, CambioVecindad)
cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 2
'''
kernel = np.array([[-1], [1]])
img = img2
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Derivada", "Filtro derivada", 1, 3, CambioVecindad)
cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 3
'''
kernel = np.array([[-1], [1]])
img = img3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Derivada", "Filtro derivada", 1, 3, CambioVecindad)
cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 4
'''
kernel = np.array([[-1], [1]])
img = img4
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada")
cv.createTrackbar("Derivada", "Filtro derivada", 1, 3, CambioVecindad)
cv.imshow("Filtro derivada", FiltroDerivada(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()