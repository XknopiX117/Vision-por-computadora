import cv2 as cv, numpy as np, matplotlib.pyplot as plt

#lectura de imagen y modificaciones
img = cv.imread("Objetos.jpeg")
img_gray = cv.imread("Objetos.jpeg", cv.IMREAD_GRAYSCALE)
th, img_th = cv.threshold(img_gray, 128, 256, cv.THRESH_BINARY)
res = cv.bitwise_and(img_gray ,img_th, mask=img_th)

#Gráfica del histograma
def Histograma():
    hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
    plt.figure("Ventana de Histograma")
    plt.title("Gráfica")
    plt.plot(hist, color='blue')
    plt.xlabel('Intensidad de iluminación')
    plt.ylabel('Cantidad de píxeles')
    plt.show()

#Mostrar imágenes
#Original
cv.namedWindow("Original")
cv.imshow("Original", img)

#En grises
cv.namedWindow("Imagen en grises")
cv.imshow("Imagen en grises", img_gray)

#Threshold
cv.namedWindow("Threshold")
cv.imshow("Threshold", img_th)

#Más brillosa
cv.namedWindow("Objeto mas brillante")
cv.imshow("Objeto mas brillante", res)

#Histograma
Histograma()

cv.waitKey(0)
cv.destroyAllWindows()