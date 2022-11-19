import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

def Operador(o):
    global kernel
    if(o == 0):

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
    elif(o == 1):

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
    elif(o == 2):

        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
    else:

        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))

def FiltroLaplaciano(image, value):
    img_l = cv.filter2D(image, -1, value)
    return img_l

'''
Imagen 1
'''
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
img = img1
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro laplaciano")
cv.createTrackbar("Operador", "Filtro laplaciano", 0, 3, Operador)
cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 2
'''
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
img = img2
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro laplaciano")
cv.createTrackbar("Operador", "Filtro laplaciano", 0, 3, Operador)
cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 3
'''
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
img = img3
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro laplaciano")
cv.createTrackbar("Operador", "Filtro laplaciano", 0, 3, Operador)
cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()

'''
Imagen 4
'''
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
img = img4
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro laplaciano")
cv.createTrackbar("Operador", "Filtro laplaciano", 0, 3, Operador)
cv.imshow("Filtro laplaciano", FiltroLaplaciano(img, kernel))
cv.waitKey(0)
cv.destroyAllWindows()