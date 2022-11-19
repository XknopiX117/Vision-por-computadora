import cv2 as cv, numpy as np; from matplotlib import pyplot as plt; import math as m

img1 = cv.imread('Ruido Periodico 1.jpg', 0)
img2 = cv.imread('Ruido Periodico 2.jpg', 0)
img3 = cv.imread('newspaper_shot_woman.tif', 0)
img4 = cv.imread('car_newsprint.tif', 0)

#Transformar a Fourier
def TransformarFourier(image):
    h, w = image.shape
    dft_M = cv.getOptimalDFTSize(w)
    dft_N = cv.getOptimalDFTSize(h)
    #Copiar
    m_aux = np.zeros((dft_N, dft_M, 2), dtype=np.float64)
    m_aux[:h, :w, 0] = image
    img_dft=cv.dft(m_aux, flags = cv.DFT_COMPLEX_OUTPUT, nonzeroRows=h)
    return img_dft

### /*INICIO FUNCIONES DE RECHAZA BANDA*\ ###
#Obtener máscara Rechaza Banda
def Mask_RechazaBanda(img_shift, vec, d1, d2, aux = 0):
    matriz = np.zeros_like(img_shift)
    if(aux == 0):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,0], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,1], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,2], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,3], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        return matriz
    elif(aux == 1):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,0], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,1], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,2], vec[0,2]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,3], vec[0,2]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,4], vec[0,4]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,5], vec[0,5]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,6], vec[0,6]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,7], vec[0,7]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        return matriz
    elif(aux == 2):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,0], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,1], vec[0,1]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,2], vec[0,2]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,3], vec[0,3]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,4], vec[0,4]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,5], vec[0,5]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,6], vec[0,6]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,7], vec[0,7]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 9
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,8], vec[0,8]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 10
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,9], vec[0,9]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 11
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,10], vec[0,10]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 12
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,11], vec[0,11]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 13
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,12], vec[0,12]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        return matriz
    elif(aux == 3):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,0], vec[0,0]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,1], vec[0,1]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,2], vec[0,2]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,3], vec[0,3]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,4], vec[0,4]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,5], vec[0,5]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,6], vec[0,6]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = Distancia((vec[1,7], vec[0,7]), (i, j))
                if (dis <= d1 and dis >= d2):
                    matriz[i, j] = 0
                else:
                    matriz[i, j] = 1
        return matriz

#Calcular distancia del circulo
def Distancia(a, b):
    dis = m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return dis
### /*FIN FUNCIONES DE RECHAZA BANDA*\ ###

### /*INICIO FUNCIONES DE BUTTERWORTH*\ ###
#Obtener máscara Butterworth
def Mask_Butterworth(img_shift, vec, d1, d2, n, aux = 0):
    matriz = np.zeros_like(img_shift)
    if(aux == 0):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        return matriz
    elif(aux == 1):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,2]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        return matriz
    elif(aux == 2):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,1]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,3]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 9
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,8], vec[0,8]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 10
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,9], vec[0,9]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 11
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,10], vec[0,10]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 12
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,11], vec[0,11]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 13
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,12], vec[0,12]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        return matriz
    elif(aux == 3):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,1]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,3]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))
        return matriz

#Calcular distancia al origen
def DistanciaOrigen(a):
    dis = m.sqrt(pow(a[0], 2)+pow(a[1],2))
    return dis

#Cambiar orden Butterworth
def CambiarOrdenButterworth(o):
    global O, r1, r2, P, A
    O = o
    mask = Mask_Butterworth(img_sdft, P, r1, r2, O, A)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
    plt.show()
### /*FIN FUNCIONES DE BUTTERWORTH*\ ###

### /*INICIO FUNCIONES DE GAUSS*\ ###
#Obtener máscara Gauss
def Mask_Gauss(img_shift, vec, d1, d2, aux = 0):
    matriz = np.zeros_like((img_shift))
    if(aux == 0):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        return matriz
    elif(aux == 1):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,2]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        return matriz
    elif(aux == 2):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,1]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,3]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 9
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,8], vec[0,8]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 10
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,9], vec[0,9]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 11
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,10], vec[0,10]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 12
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,11], vec[0,11]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 13
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,12], vec[0,12]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        return matriz
    elif(aux == 3):
        #Punto 1
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,0], vec[0,0]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 2
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,1], vec[0,1]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 3
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,2], vec[0,2]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 4
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,3], vec[0,3]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,4], vec[0,4]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 6
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,5], vec[0,5]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 7
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,6], vec[0,6]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        #Punto 8
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                dis = DistanciaOrigen((vec[1,7], vec[0,7]))
                matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))
        return matriz
### img1 ###
'''
Mostrar filtro rechaza banda
'''
img = img1
P = np.array([[97, 97, 97, 97],[188, 70, 11, 129]])
r1, r2 = 7, 5
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_RechazaBanda(img_sdft, P, r1, r2)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Butterworth
'''
A = 0
P = np.array([[97, 97, 97, 97],[188, 70, 11, 129]])
r1, r2 = 7, 5
cv.namedWindow("Original")
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
O = 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, P, r1, r2, O)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Gauss
'''
P = np.array([[97, 97, 97, 97],[188, 70, 11, 129]])
r1, r2 = 10, 7
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, P, r1, r2)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

### img2 ###
'''
Mostrar filtro rechaza banda
'''
img = img2
P = np.array([[75.5, 75.5, 182.5, 182.5, 54, 33, 225, 204],[43.5, 150.5, 43.5, 150.5, 97, 97, 97, 97]])
r1, r2 = 4, 3
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_RechazaBanda(img_sdft, P, r1, r2, 1)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Butterworth
'''
A = 1
P = np.array([[75.5, 75.5, 182.5, 182.5, 54, 33, 225, 204],[43.5, 150.5, 43.5, 150.5, 97, 97, 97, 97]])
r1, r2 = 4, 3
cv.namedWindow("Original")
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
O = 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, P, r1, r2, O, A)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Gauss
'''
P = np.array([[75.5, 75.5, 182.5, 182.5, 54, 33, 225, 204],[43.5, 150.5, 43.5, 150.5, 97, 97, 97, 97]])
r1, r2 = 12, 9
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, P, r1, r2, 1)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

### img3 ###
'''
Mostrar filtro rechaza banda
'''
img = img3
P = np.array([[249, 249, 288, 288, 366, 366, 444, 444, 287, 326, 326, 404, 404, 443],
                [549, 369, 325, 503, 503, 325, 325, 504, 414, 460, 370, 370, 460, 414]])
r1, r2 = 11, 25
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_RechazaBanda(img_sdft, P, r1, r2, 2)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Butterworth
'''
A = 2
P = np.array([[249, 249, 288, 288, 366, 366, 444, 444, 287, 326, 326, 404, 404, 443],
                [549, 369, 325, 503, 503, 325, 325, 504, 414, 460, 370, 370, 460, 414]])
r1, r2 = 11, 25
cv.namedWindow("Original")
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
O = 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, P, r1, r2, O, A)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Gauss
'''
P = np.array([[249, 249, 288, 288, 366, 366, 444, 444, 287, 326, 326, 404, 404, 443],
                [549, 369, 325, 503, 503, 325, 325, 504, 414, 460, 370, 370, 460, 414]])
r1, r2 = 11, 25
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, P, r1, r2, 2)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

### img4 ###
'''
Mostrar filtro rechaza banda
'''
img = img4
P = np.array([[52, 113, 54, 110, 56, 113, 57, 112], 
                [44, 41, 84, 80, 165, 162, 207, 204]])
r1, r2 = 14, 10
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_RechazaBanda(img_sdft, P, r1, r2, 3)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Butterworth
'''
A = 3
P = np.array([[52, 113, 54, 110, 56, 113, 57, 112], 
                [44, 41, 84, 80, 165, 162, 207, 204]])
r1, r2 = 14, 10
cv.namedWindow("Original")
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
O = 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, P, r1, r2, O, A)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Gauss
'''
P = np.array([[52, 113, 54, 110, 56, 113, 57, 112], 
                [44, 41, 84, 80, 165, 162, 207, 204]])
r1, r2 = 14, 10
cv.namedWindow("Original")
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, P, r1, r2, 3)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()