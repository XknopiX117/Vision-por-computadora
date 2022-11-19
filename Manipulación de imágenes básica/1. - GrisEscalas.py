import cv2
import numpy as np

#Cargar imagenes y escala de grises de la imágen
Imagen = cv2.imread("gir.jpg")
Imagen_Gris = cv2.imread("gir.jpg", 0)

# Cambiar Tamaño usando OpenCV
# cv.INTER_NEAREST
# cv.INTER_LINEAR
# cv.INTER_CUBIC
# cv.INTER_AREA
# cv.INTER_LANCZOS4
# cv.INTER_LINEAR_EXACT

#Resolución 1.5
Img1 = cv2.resize(Imagen_Gris, (0,0), fx = 1.5, fy = 1.5, interpolation = cv2.INTER_NEAREST)
Img2 = cv2.resize(Imagen_Gris, (0,0), fx = 1.5, fy = 1.5, interpolation = cv2.INTER_LINEAR)
Img3 = cv2.resize(Imagen_Gris, (0,0), fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)

cv2.imshow("Imagen original", Imagen)
cv2.imshow("Imagen en gris", Imagen_Gris)
cv2.imshow("Vecino mas cercano", Img1)
cv2.imshow("Bilineal", Img2)
cv2.imshow("Bicubica", Img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Resolución 0.5
Img1 = cv2.resize(Imagen_Gris, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)
Img2 = cv2.resize(Imagen_Gris, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR)
Img3 = cv2.resize(Imagen_Gris, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)

cv2.imshow("Imagen original", Imagen)
cv2.imshow("Imagen en gris", Imagen_Gris)
cv2.imshow("Vecino mas cercano", Img1)
cv2.imshow("Bilineal", Img2)
cv2.imshow("Bicubica", Img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Resolución 0.25
Img1 = cv2.resize(Imagen_Gris, (0,0), fx = 0.25, fy = 0.25, interpolation = cv2.INTER_NEAREST)
Img2 = cv2.resize(Imagen_Gris, (0,0), fx = 0.25, fy = 0.25, interpolation = cv2.INTER_LINEAR)
Img3 = cv2.resize(Imagen_Gris, (0,0), fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)

cv2.imshow("Imagen original", Imagen)
cv2.imshow("Imagen en gris", Imagen_Gris)
cv2.imshow("Vecino mas cercano", Img1)
cv2.imshow("Bilineal", Img2)
cv2.imshow("Bicubica", Img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Resolución 0.125
Img1 = cv2.resize(Imagen_Gris, (0,0), fx = 0.125, fy = 0.125, interpolation = cv2.INTER_NEAREST)
Img2 = cv2.resize(Imagen_Gris, (0,0), fx = 0.125, fy = 0.125, interpolation = cv2.INTER_LINEAR)
Img3 = cv2.resize(Imagen_Gris, (0,0), fx = 0.125, fy = 0.125, interpolation = cv2.INTER_CUBIC)

cv2.imshow("Imagen original", Imagen)
cv2.imshow("Imagen en gris", Imagen_Gris)
cv2.imshow("Vecino mas cercano", Img1)
cv2.imshow("Bilineal", Img2)
cv2.imshow("Bicubica", Img3)
cv2.waitKey(0)
cv2.destroyAllWindows()