import cv2 as cv, numpy as np; from matplotlib import pyplot as plt; import math as m

img = cv.imread('Lena.tif',cv.IMREAD_GRAYSCALE)

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
### /*INICIO FUNCIONES DE PASA ALTA*\ ###
#Obtener máscara Pasa Alta
def Mask_PasaAlta(img_shift, d):
    matriz = np.zeros_like(img_shift)
    matriz_aux = matriz
    centro = tuple(map(lambda x : (x-1)/2, matriz.shape[:2]))
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = Distancia(centro, (i, j))
            if dis <= d:
                matriz[i, j] = 1
                matriz_aux[i, j] = 1 - matriz[i, j]
            else:
                matriz[i, j] = 0
                matriz_aux[i, j] = 1 - matriz[i, j]

    return matriz_aux

#Calcular distancia del circulo
def Distancia(a, b):
    dis = m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return dis

#cambiar la distancia
def CambiarDistancia_PasaAlta(d):
    global D, mask
    D = d
    mask = Mask_PasaAlta(img_sdft, D)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Pasa alta'), plt.xticks([]), plt.yticks([])
    plt.show()
### /*FIN FUNCIONES DE PASA BAJA*\ ###

### /*INICIO FUNCIONES DE BUTTERWORTH*\ ###
#Obtener máscara Butterworth
def Mask_Butterworth(img_shift, d, n):
    matriz = np.zeros_like(img_shift)
    matriz_aux = matriz
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = DistanciaOrigen((i, j))
            matriz[i, j] = 1/(1+((dis/d)**(2*n)))
            matriz_aux[i, j] = 1 - matriz[i, j]

    return matriz_aux

#Calcular distancia al origen
def DistanciaOrigen(a):
    dis = m.sqrt(pow(a[0], 2)+pow(a[1],2))
    return dis

#Cambiar orden Butterworth
def CambiarOrdenButterworth(o):
    global O
    O = o
    mask = Mask_Butterworth(img_sdft, D, O)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
    plt.show()

#cambiar la distancia
def CambiarDistancia_Butterworth(d):
    global D, mask, O
    D = d
    mask = Mask_Butterworth(img_sdft, D, O)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
    plt.show()
### /*FIN FUNCIONES DE BUTTERWORTH*\ ###

### /*INICIO FUNCIONES DE GAUSS*\ ###
#Obtener máscara Gauss
def Mask_Gauss(img_shift, d):
    matriz = np.zeros_like(img_shift)
    matriz_aux = matriz
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = DistanciaOrigen((i, j))
            matriz[i, j] = np.power(np.e, np.divide(-np.power(dis,2), 2*np.power(d,2)))
            matriz_aux[i, j] = 1 - matriz[i, j]

    return matriz_aux

#cambiar la distancia
def CambiarDistancia_Gauss(d):
    global D, mask
    D = d + 1
    mask = Mask_Gauss(img_sdft, D)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
    plt.show()

'''
Mostrar filtro pasa alta
'''
D = 50
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia", "Original", 50, 150, CambiarDistancia_PasaAlta)
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_PasaAlta(img_sdft, D)
img_mask = img_sdft * mask
img_isdft = np.fft.ifftshift(img_mask)
img_filtrada = cv.idft(img_isdft)
img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
plt.imshow(img_filtrada, cmap = 'gray')
plt.title('Imagen filtrada Pasa alta'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

'''
Mostrar Filtro Butterworth
'''
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia", "Original", 50, 300, CambiarDistancia_Butterworth)
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
D, O = 50, 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, D, O)
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
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia", "Original", 49, 200, CambiarDistancia_Gauss)
cv.imshow("Original", img)
D = 50
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, D)
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