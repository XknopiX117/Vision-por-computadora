import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

#Canny normal
def CambioThreshold1(t1):
    global T1
    T1 = t1*10
    cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))

def CambioThreshold2(t2):
    global T2
    T2 = t2*10
    cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))

def FiltroCanny(image, t1, t2):
    img_canny = cv.Canny(image, t1, t2)
    return img_canny

#Gaus Canny
def CambioGThreshold1(t1):
    global G1
    G1 = t1*10
    cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))

def CambioGThreshold2(t2):
    global G2
    G2 = t2*10
    cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))

def CambioVecindad(c):
    global kernel
    kernel = 2*c+3
    cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))

def FiltroGCanny(image, t1, t2, var):
    img_gauss = cv.GaussianBlur(image, (var, var), 0)
    img_gcanny = cv.Canny(img_gauss, t1, t2)
    return img_gcanny

'''
Imagen 1
'''
T1, T2 = 0, 0
G1, G2 = 0, 0
img = img1
kernel = 3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro Canny")
cv.createTrackbar("Umbral 1", "Filtro Canny", 0, 100, CambioThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Canny", 0, 100, CambioThreshold2)
cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))
cv.namedWindow("Filtro Gauss Canny")
cv.createTrackbar("Umbral 1", "Filtro Gauss Canny", 0, 100, CambioGThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Gauss Canny", 0, 100, CambioGThreshold2)
cv.createTrackbar("Vecindad", "Filtro Gauss Canny", 0, 23, CambioVecindad)
cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 2
'''
T1, T2 = 0, 0
G1, G2 = 0, 0
img = img2
kernel = 3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro Canny")
cv.createTrackbar("Umbral 1", "Filtro Canny", 0, 100, CambioThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Canny", 0, 100, CambioThreshold2)
cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))
cv.namedWindow("Filtro Gauss Canny")
cv.createTrackbar("Umbral 1", "Filtro Gauss Canny", 0, 100, CambioGThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Gauss Canny", 0, 100, CambioGThreshold2)
cv.createTrackbar("Vecindad", "Filtro Gauss Canny", 0, 23, CambioVecindad)
cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 3
'''
T1, T2 = 0, 0
G1, G2 = 0, 0
img = img3
kernel = 3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro Canny")
cv.createTrackbar("Umbral 1", "Filtro Canny", 0, 100, CambioThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Canny", 0, 100, CambioThreshold2)
cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))
cv.namedWindow("Filtro Gauss Canny")
cv.createTrackbar("Umbral 1", "Filtro Gauss Canny", 0, 100, CambioGThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Gauss Canny", 0, 100, CambioGThreshold2)
cv.createTrackbar("Vecindad", "Filtro Gauss Canny", 0, 23, CambioVecindad)
cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 4
'''
T1, T2 = 0, 0
G1, G2 = 0, 0
img = img4
kernel = 3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro Canny")
cv.createTrackbar("Umbral 1", "Filtro Canny", 0, 100, CambioThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Canny", 0, 100, CambioThreshold2)
cv.imshow("Filtro Canny", FiltroCanny(img, T1, T2))
cv.namedWindow("Filtro Gauss Canny")
cv.createTrackbar("Umbral 1", "Filtro Gauss Canny", 0, 100, CambioGThreshold1)
cv.createTrackbar("Umbral 2", "Filtro Gauss Canny", 0, 100, CambioGThreshold2)
cv.createTrackbar("Vecindad", "Filtro Gauss Canny", 0, 23, CambioVecindad)
cv.imshow("Filtro Gauss Canny", FiltroGCanny(img, G1, G2, kernel))
cv.waitKey(0)
cv.destroyAllWindows()