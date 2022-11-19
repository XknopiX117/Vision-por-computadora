import cv2 as cv
import numpy as np

img1 = cv.imread('Juguetes de Colores.jpg')
img2 = cv.imread('Figuras de Colores.jpg')

img1 = cv.resize(img1, (0,0), fx = 0.2, fy = 0.2, interpolation=cv.INTER_CUBIC)
img2 = cv.resize(img2, (0,0), fx = 0.5, fy = 0.5, interpolation=cv.INTER_CUBIC)

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img1_hls = cv.cvtColor(img1, cv.COLOR_BGR2HLS)
img2_hls = cv.cvtColor(img2, cv.COLOR_BGR2HLS)

#RGB
UvaRGB_max = np.array([178, 66, 255])
UvaRGB_min = np.array([78, 29, 112])
NaranjaRGB_max = np.array([65, 205, 245])
NaranjaRGB_min = np.array([0, 60, 240])

#HLS
UvaHLS_max = np.array([166, 255, 255])
UvaHLS_min = np.array([126, 0, 41])
NaranjaHLS_max = np.array([22, 172, 255])
NaranjaHLS_min = np.array([17, 120, 196])

def Extraccion(image, list_max, list_min):
    mask = cv.inRange(image, list_min, list_max)
    ext = cv.bitwise_and(image, image, mask=mask)
    return ext

#Mostrar
cv.imshow("Original 1", img1)
cv.imshow("RGB 1", img1_rgb)
cv.imshow("HLS 1", img1_hls)
cv.imshow("Segmentacion RGB 1", Extraccion(img1, UvaRGB_max, UvaRGB_min))
cv.imshow("Segmentacion HLS 1", Extraccion(img1_hls, UvaHLS_max, UvaHLS_min))
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("Original 2", img2)
cv.imshow("RGB 2", img2_rgb)
cv.imshow("HLS 2", img2_hls)
cv.imshow("Segmentacion RGB 2", Extraccion(img2, NaranjaRGB_max, NaranjaRGB_min))
cv.imshow("Segmentacion HLS 2", Extraccion(img2_hls, NaranjaHLS_max, NaranjaHLS_min))
cv.waitKey(0)
cv.destroyAllWindows()