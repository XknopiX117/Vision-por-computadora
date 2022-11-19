import cv2 as cv
import numpy as np
import pywt as pw
import matplotlib.pyplot as plt

img = 0
img1 = cv.imread('Ruido Periodico 1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Valle de la luna.jpg', cv.IMREAD_GRAYSCALE)

def Wavelet(image, wavelet_name, sigma):
    new_coeffs = []
    wavelet = pw.Wavelet(wavelet_name)
    coeffs = pw.wavedec2(image[:, :], wavelet)
    threshold = sigma*np.sqrt(2*np.log2(image.size))

    for i in range(len(coeffs) - 1):
        if (i == 0):
            c = pw.threshold(coeffs[0], threshold)
            new_coeffs.append(c)
        else:
            c0 = pw.threshold(coeffs[i][0], threshold)
            c1 = pw.threshold(coeffs[i][1], threshold)
            c2 = pw.threshold(coeffs[i][2], threshold)
            new_coeffs.append((c0, c1, c2))

    new_img = pw.waverec2(new_coeffs, wavelet)
    new_img = np.multiply(np.divide(new_img - np.min(new_img),(np.max(new_img) - np.min(new_img))), 255)
    new_img = new_img.astype(np.uint8)
    return new_img, wavelet_name

'''Imagen 1'''
img = img1
#Wavelet Haar
img_denoise, name = Wavelet(img, 'gauss2', 20)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)

#Wavelet Daub2
img_denoise, name = Wavelet(img, 'db2', 27)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)

#Wavelet Daub4
img_denoise, name = Wavelet(img, 'db4', 30)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)

'''Imagen 2'''
img = img2
#Wavelet Haar
img_denoise, name = Wavelet(img, 'haar', 10)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)

#Wavelet Daub2
img_denoise, name = Wavelet(img, 'db2', 10)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)

#Wavelet Daub4
img_denoise, name = Wavelet(img, 'db4', 10)
plt.title(name)
plt.imshow(img_denoise, 'gray')
plt.show()
cv.waitKey(0)