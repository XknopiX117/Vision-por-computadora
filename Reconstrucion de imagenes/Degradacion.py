import cv2 as cv 
import numpy as np

img = cv.imread("linkRuido.jpg", cv.IMREAD_GRAYSCALE)

h, w = img.shape

img_f = np.float32(img)
img_fft = np.fft.fft2(img_f)
img_sfft = np.fft.fftshift(img_fft)

mask = np.copy(img_sfft)
cy, cx = h//2, w//2
a, b, T, aux = 0.15, 0.2, 1, 0.0
for y in range(h):
    for x in range(w):
        aux = np.pi*((y-cy)*a+(x-cx)*b)
        if aux == 0:
            aux = 0.001
        mask[y, x] = (T/aux)*np.sin(aux)*np.e**(-1j*aux)

img_mask = img_sfft*mask
img_isfft = np.fft.ifftshift(img_mask)
img_ifft = np.fft.ifft2(img_isfft)
img_deg = np.log(np.absolute(img_ifft))
img_deg = (img_deg-np.amin(img_deg))/(np.amax(img_deg)-np.amin(img_deg))
cv.imshow("Degradacion", img_deg)
cv.waitKey(0)
cv.destroyAllWindows()