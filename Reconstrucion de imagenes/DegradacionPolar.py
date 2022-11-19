import cv2 as cv 
import numpy as np

img = cv.imread("linkRuido.jpg", cv.IMREAD_GRAYSCALE)

h, w = img.shape

rad_max = np.sqrt(np.power(h,2)/2+np.power(w,2)/2)
img_polar = cv.linearPolar(img, (w/2, h/2), rad_max, cv.WARP_FILL_OUTLIERS)
img_f = np.float32(img_polar)
img_fft = np.fft.fft2(img_f)
img_sfft = np.fft.fftshift(img_fft)

H = np.copy(img_sfft)
t, r = img_polar.shape
ct, cr = t//2, r//2
a, b, T, aux = 0.02, 0.02, 1, 0.0
for th in range(t):
    for rad in range(r):
        aux = np.pi*((th-ct)*a+(rad-cr)*b)
        if aux == 0:
            aux = 0.001
        H[th, rad] = (T/aux)*np.sin(aux)*np.e**(-1j*aux)

G = img_sfft * H
img_isfft = np.fft.ifftshift(G)
img_ifft = np.fft.ifft2(img_isfft)
img_deg = np.log(np.absolute(img_ifft))
img_deg = (img_deg-np.amin(img_deg))/(np.amax(img_deg)-np.amin(img_deg))
img_deg = cv.linearPolar(img_deg, (w/2, h/2), rad_max, cv.WARP_INVERSE_MAP)
cv.imshow("Degradacion Lineal y Rotacional", img_deg)
cv.waitKey(0)
cv.destroyAllWindows()