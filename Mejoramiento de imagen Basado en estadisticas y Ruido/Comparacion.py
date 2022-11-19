from typing import Mapping
import cv2 as cv, numpy as np, glob as g, matplotlib.pyplot as plt

#Leer imagenes
img = []
path = g.glob("C:/Users/jeres/Desktop/Mejoramiento de imagen Basado en estadisticas y Ruido/Imagenes_pruebas/*")
for file in path:

    f = cv.imread(file, 0)
    img.append(f)
    
#Crear histograma
def Histograma():
    hist_n = cv.calcHist([nueva_img], [0], None, [256], [0, 256])
    hist_o = cv.calcHist([comp], [0], None, [256], [0, 256])
    plt.figure("Ventana de Histogramas")
    plt.title("Gráfica")
    plt.plot(hist_n, color='blue', label = 'Histograma propio')
    plt.plot(hist_o, color = "red", label = 'Histograma OpenCV')
    plt.xlabel('Intensidad de iluminación')
    plt.ylabel('Cantidad de píxeles')
    plt.legend()
    plt.show()

#Slider para cambiar de imagen
def Cambio(c):
    global S
    S = c
    plt.clf()
    cv.imshow("Imagen", img[S])
    Mejorar()
    Comparacion()
    Histograma()

#Ecualizador de histograma
def Mejorar():
    global nueva_img
    histograma, bins = np.histogram(img[S].flatten(), 256, [0, 256])
    acumulada = histograma.cumsum() #hacemos la sumatoria del hostograma
    acumulada_n = acumulada * float(histograma.max()) / acumulada.max() #normalizamos
    acumulada_m = np.ma.masked_equal(acumulada, 0) #masked
    acumulada_m = (acumulada_m - acumulada_m.min())*255/(acumulada_m.max()-acumulada_m.min())
    final = np.ma.filled(acumulada_m, 0).astype('uint8')
    nueva_img = final[img[S]]
    cv.imshow("Nueva con histograma personal", nueva_img)

def Comparacion():
    global comp
    comp = cv.equalizeHist(img[S])
    cv.imshow("Comparacion", comp)

S = 0
nueva_img = 0
comp = 0
cv.namedWindow("Imagen")
cv.imshow("Imagen", img[S])
cv.createTrackbar("Seleccion", "Imagen", 0, 6, Cambio)
cv.namedWindow("Nueva con histograma personal")
Mejorar()
cv.namedWindow("Comparacion")
Comparacion()
Histograma()
cv.waitKey(0)
cv.destroyAllWindows()
