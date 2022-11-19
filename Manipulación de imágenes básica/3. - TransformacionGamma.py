import cv2 as cv, numpy as np

#Cargar las imágenes
img1 = cv.imread('BajoContrasteGrises.jpg')
img2 = cv.imread('SobreeExpuestaGrises.jpg')
img3 = cv.imread('SubexpuestaGrises.jpg')
img = [img1, img2, img3]

#Cambiar de imagen
def Cambiar(sel):
    global S
    S = sel
    cv.imshow("Imagen", img[S])

#Función Gamma
def FuncionG(g):
    gamma = g/10
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img[S], lookUpTable)
    cv.imshow("Imagen", res)

#Cargar ventana inicial
S = 0
cv.namedWindow('Imagen')
cv.imshow("Imagen", img[S])
cv.createTrackbar("Seleccion", "Imagen", 0, 2, Cambiar)
cv.createTrackbar("Gamma", "Imagen", 20, 40, FuncionG)
cv.waitKey(0)
cv.destroyAllWindows()