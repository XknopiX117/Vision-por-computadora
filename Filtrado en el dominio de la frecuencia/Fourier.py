import cv2 as cv, numpy as np; from matplotlib import pyplot as plt; import math as m

img1 = cv.imread('Ruido Periodico 1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('Ruido Periodico 2.jpg', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('newspaper_shot_woman.tif', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('car_newsprint.tif', cv.IMREAD_GRAYSCALE)

img = img1
f_img = np.fft.fft2(img)
sf_img = np.fft.fftshift(f_img)
p = np.abs(sf_img)
q = np.log(p)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(q, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])
plt.show()

img = img2
f_img = np.fft.fft2(img)
sf_img = np.fft.fftshift(f_img)
p = np.abs(sf_img)
q = np.log(p)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(q, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])
plt.show()

img = img3
f_img = np.fft.fft2(img)
sf_img = np.fft.fftshift(f_img)
p = np.abs(sf_img)
q = np.log(p)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(q, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])
plt.show()

img = img4
f_img = np.fft.fft2(img)
sf_img = np.fft.fftshift(f_img)
p = np.abs(sf_img)
q = np.log(p)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(q, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])
plt.show()