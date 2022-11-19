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
    hist_a = cv.calcHist([img[0]], [0], None, [256], [0, 256])
    hist_n = cv.calcHist([nueva_img], [0], None, [256], [0, 256])
    plt.figure("Ventana de Histogramas")
    plt.title("Gráfica")
    plt.plot(hist_n, color='blue', label = 'Histograma nuevo')
    plt.plot(hist_a, color = "red", label = 'Histograma anterior')
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
    Histograma()

#Ecualizador de histograma
def Mejorar():
    global nueva_img
    hist, bins = np.histogram(img[S].flatten(), 256, [0, 256])
    acum = hist.cumsum() #Sumatoria acumulada
    acum_m = np.ma.masked_equal(acum, 0) #Mask
    acum_m = (acum_m - acum_m.min())*255/(acum_m.max()-acum_m.min()) #Normalizacion
    final = np.ma.filled(acum_m, 0).astype('uint8')
    nueva_img = final[img[S]]
    cv.imshow("Nueva", nueva_img)

S = 0
nueva_img = 0
cv.namedWindow("Imagen")
cv.imshow("Imagen", img[S])
cv.createTrackbar("Seleccion", "Imagen", 0, 6, Cambio)
cv.namedWindow("Nueva")
Mejorar()
Histograma()
cv.waitKey(0)
cv.destroyAllWindows()