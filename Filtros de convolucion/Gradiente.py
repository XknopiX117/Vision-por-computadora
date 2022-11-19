import cv2 as cv, numpy as np

img1 = cv.imread("lena_gray_512.tif" ,cv.IMREAD_GRAYSCALE)
img2 = cv.imread("livingroom.tif" ,cv.IMREAD_GRAYSCALE)
img3 = cv.imread("mandril_gray.tif" ,cv.IMREAD_GRAYSCALE)
img4 = cv.imread("walkbridge.tif" ,cv.IMREAD_GRAYSCALE)
global img

def Operador(o):
    global Op
    global kernelx, kernely
    Op = o
    if(o == 0):
        cv.imshow("Filtro derivada x", FiltroSobelx(img))
        cv.imshow("Filtro derivada y", FiltroSobely(img))
        cv.imshow("Gradiente", Gradiente(img))
    else:
        kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        cv.imshow("Filtro derivada x", FiltroPrewittx(img, kernelx))
        cv.imshow("Filtro derivada y", FiltroPrewitty(img, kernely))
        cv.imshow("Gradiente", Gradiente(img))

def FiltroSobelx(image):
    img_derx = cv.Sobel(image, -1, 1, 0)
    return img_derx

def FiltroSobely(image):
    img_dery = cv.Sobel(image, -1, 0, 1)
    return img_dery

def FiltroPrewittx(image, matriz1):
    img_derx = cv.filter2D(image, -1, matriz1)
    return img_derx

def FiltroPrewitty(image, matriz2):
    img_dery = cv.filter2D(image, -1, matriz2)
    return img_dery

def Gradiente(image):
    if(Op == 0):
        return (FiltroSobelx(image)**2+FiltroSobely(image)**2)**1/2
    else:
        return (FiltroPrewittx(image, kernelx)**2+FiltroPrewitty(image, kernely)**2)**1/2

'''
Imagen 1
'''
img = img1
der = 0
Op = 0
kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img)
cv.namedWindow("Filtro derivada x")
cv.namedWindow("Filtro derivada y")
cv.namedWindow("Gradiente")
cv.createTrackbar("Operador", "Imagen original", 0, 1, Operador)
cv.imshow("Filtro derivada x", FiltroSobelx(img))
cv.imshow("Filtro derivada y", FiltroSobely(img))
cv.imshow("Gradiente", Gradiente(img))
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
cv.namedWindow("Filtro derivada x")
cv.namedWindow("Filtro derivada y")
cv.namedWindow("Gradiente")
cv.createTrackbar("Operador", "Imagen original", 0, 1, Operador)
cv.imshow("Filtro derivada x", FiltroSobelx(img))
cv.imshow("Filtro derivada y", FiltroSobely(img))
cv.imshow("Gradiente", Gradiente(img))
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
cv.namedWindow("Filtro derivada x")
cv.namedWindow("Filtro derivada y")
cv.namedWindow("Gradiente")
cv.createTrackbar("Operador", "Imagen original", 0, 1, Operador)
cv.imshow("Filtro derivada x", FiltroSobelx(img))
cv.imshow("Filtro derivada y", FiltroSobely(img))
cv.imshow("Gradiente", Gradiente(img))
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
cv.namedWindow("Filtro derivada x")
cv.namedWindow("Filtro derivada y")
cv.namedWindow("Gradiente")
cv.createTrackbar("Operador", "Imagen original", 0, 1, Operador)
cv.imshow("Filtro derivada x", FiltroSobelx(img))
cv.imshow("Filtro derivada y", FiltroSobely(img))
cv.imshow("Gradiente", Gradiente(img))
cv.waitKey(0)
cv.destroyAllWindows()