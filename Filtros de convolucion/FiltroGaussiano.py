import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

def CambioVecindad(c):
    global kernel
    kernel = 2*c+3
    cv.imshow("Filtro gaussiano", FiltroGaussiano(img, kernel))

def FiltroGaussiano(image, matriz):
    img_prom = cv.GaussianBlur(image, (matriz, matriz), 0)
    return img_prom

'''
Imagen 1
'''
kernel = 3
img = img1
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro gaussiano")
cv.createTrackbar("Vecindad", "Filtro gaussiano", 0, 23, CambioVecindad)
cv.imshow("Filtro gaussiano", FiltroGaussiano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 2
'''
kernel = 3
img = img2
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro gaussiano")
cv.createTrackbar("Vecindad", "Filtro gaussiano", 0, 23, CambioVecindad)
cv.imshow("Filtro gaussiano", FiltroGaussiano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 3
'''
kernel = 3
img = img3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro gaussiano")
cv.createTrackbar("Vecindad", "Filtro gaussiano", 0, 23, CambioVecindad)
cv.imshow("Filtro gaussiano", FiltroGaussiano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 4
'''
kernel = 3
img = img4
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro gaussiano")
cv.createTrackbar("Vecindad", "Filtro gaussiano", 0, 23, CambioVecindad)
cv.imshow("Filtro gaussiano", FiltroGaussiano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()