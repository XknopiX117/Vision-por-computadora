import cv2 as cv, numpy as np; from matplotlib import pyplot as plt; import math as m

img = cv.imread('Lena.tif', cv.IMREAD_GRAYSCALE)

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
def Mask_PasaBanda(img_shift, d1, d2):
    matriz = np.zeros_like(img_shift)
    centro = tuple(map(lambda x : (x-1)/2, matriz.shape[:2]))
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = Distancia(centro, (i, j))
            if dis <= d1 and dis >= d2:
                matriz[i, j] = 0
            else:
                matriz[i, j] = 1

    return matriz

#Calcular distancia del circulo
def Distancia(a, b):
    dis = m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return dis

#cambiar la distancia
def CambiarDistancia1_PasaBanda(d):
    global D1, D2, mask
    D1 = d
    mask = Mask_PasaBanda(img_sdft, D1, D2)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
    plt.show()

def CambiarDistancia2_PasaBanda(d):
    global D1, D2, mask
    D2 = d
    mask = Mask_PasaBanda(img_sdft, D1, D2)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Rechaza banda'), plt.xticks([]), plt.yticks([])
    plt.show()
### /*FIN FUNCIONES DE PASA BAJA*\ ###

### /*INICIO FUNCIONES DE BUTTERWORTH*\ ###
#Obtener máscara Butterworth
def Mask_Butterworth(img_shift, d1, d2, n):
    matriz = np.zeros_like(img_shift)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = DistanciaOrigen((i, j))
            matriz[i, j] = 1 - ((1/(1+((dis/d1)**(2*n))))*(1 - 1/(1+((dis/d2)**(2*n)))))

    return matriz

#Calcular distancia al origen
def DistanciaOrigen(a):
    dis = m.sqrt(pow(a[0], 2)+pow(a[1],2))
    return dis

#Cambiar orden Butterworth
def CambiarOrdenButterworth(o):
    global O, D1, D2
    O = o
    mask = Mask_Butterworth(img_sdft, D1, D2, O)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada-np.amin(img_filtrada))/(np.amax(img_filtrada)-np.amin(img_filtrada))
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
    plt.show()

#cambiar la distancia
def CambiarDistancia1_Butterworth(d):
    global D1, D2, mask, O
    D1 = d
    mask = Mask_Butterworth(img_sdft, D1, D2, O)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Butterworth'), plt.xticks([]), plt.yticks([])
    plt.show()

def CambiarDistancia2_Butterworth(d):
    global D1, D2, mask, O
    D2 = d
    mask = Mask_Butterworth(img_sdft, D1, D2, O)
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
def Mask_Gauss(img_shift, d1, d2):
    matriz = np.zeros_like(img_shift)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            dis = DistanciaOrigen((i, j))
            matriz[i, j] = 1 - ((np.exp((-dis**2)/(2*(d1**2)))) * (1 - np.exp((-dis**2)/(2*(d2**2)))))

    return matriz

#cambiar la distancia
def CambiarDistancia1_Gauss(d):
    global D1, D2, mask
    D1 = d + 1
    mask = Mask_Gauss(img_sdft, D1, D2)
    img_mask = img_sdft * mask
    img_isdft = np.fft.ifftshift(img_mask)
    img_filtrada = cv.idft(img_isdft)
    img_filtrada = cv.magnitude (img_filtrada[:,:, 0], img_filtrada[:,:, 1])
    img_filtrada = (img_filtrada - np.amin(img_filtrada))/(np.amax(img_filtrada) - np.amin(img_filtrada))
    #Mostrar filtro
    plt.imshow(img_filtrada, cmap = 'gray')
    plt.title('Imagen filtrada Gauss'), plt.xticks([]), plt.yticks([])
    plt.show()

def CambiarDistancia2_Gauss(d):
    global D1, D2, mask
    D2 = d + 1
    mask = Mask_Gauss(img_sdft, D1, D2)
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
D1, D2 = 50, 50
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia 1", "Original", 50, 150, CambiarDistancia1_PasaBanda)
cv.createTrackbar("Frecuencia 2", "Original", 50, 150, CambiarDistancia2_PasaBanda)
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_PasaBanda(img_sdft, D1, D2)
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
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia 1", "Original", 50, 300, CambiarDistancia1_Butterworth)
cv.createTrackbar("Frecuencia 2", "Original", 50, 300, CambiarDistancia2_Butterworth)
cv.createTrackbar("Orden", "Original", 0, 10, CambiarOrdenButterworth)
cv.imshow("Original", img)
D1, D2, O = 50, 50, 0
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Butterworth(img_sdft, D1, D2, O)
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
D1, D2 = 50, 50
cv.namedWindow("Original")
cv.createTrackbar("Frecuencia 1", "Original", 49, 400, CambiarDistancia1_Gauss)
cv.createTrackbar("Frecuencia 2", "Original", 49, 400, CambiarDistancia2_Gauss)
cv.imshow("Original", img)
img_dft = TransformarFourier(img)
img_sdft = np.fft.fftshift(img_dft)
mask = Mask_Gauss(img_sdft, D1, D2)
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