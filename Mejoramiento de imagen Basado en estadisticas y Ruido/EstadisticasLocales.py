import cv2 as cv, numpy as np, glob as g

#Copiar imagen
img = []
path = g.glob("C:/Users/jeres/Desktop/Mejoramiento de imagen Basado en estadisticas y Ruido/Imagenes_pruebas/*")
for file in path:
    f = cv.imread(file, 0)
    img.append(f)

#Slider para cambiar de imagen
def Cambio(c):
    global S
    S = c
    cv.imshow("Imagen original", img[S])
    Mejorar(img[S], 3, 1, 0.02, 0.8, 0.75)

def Mejorar(image, window, k0, k1, k2, E):

    #Dimensiones de img y ventana
    M, N = image.shape
    l = int((window - 1)/2)

    # Media y desviaciones
    m_g = image.mean()
    d_g = image.std()
    m_l, d_l = 0, 0

    img_nueva = np.zeros((M - 2*l, N - 2*l), np.uint8)

    img_local = np.zeros((window, window), np.uint8)

    for i in range(l, M - l):
        for j in range(l, N - l):

            #Vecidad
            for x in range(-l, l + 1):
                for y in range(-l, l + 1):
                    img_local[x + l, y + l] = image[i + x, j + y]

            # Calcular la media y desviación de la vecindad
            m_l = img_local.mean()
            d_l = img_local.std()

            #Evaluación
            if(m_l <= k0*m_g and k1*d_g <= d_l <= k2*d_g):
                if(E*image[i, j] > 255):
                    img_nueva[i - l, j - l] = 255
                else:
                    img_nueva[i - l, j - l] = round(E*image[i, j])     
            else:
                img_nueva[i - l, j - l] = image[i, j]

    cv.imshow("Mejora", img_nueva)

S = 0
cv.namedWindow("Imagen original")
cv.imshow("Imagen original", img[0])
cv.createTrackbar("Seleccion", "Imagen original", 0, 6, Cambio)
cv.namedWindow("Mejora")
Mejorar(img[S], 3, 1, 0.02, 0.8, 0.75)

cv.waitKey(0)
cv.destroyAllWindows()