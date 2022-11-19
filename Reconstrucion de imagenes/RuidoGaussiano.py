import cv2 as cv, numpy as np

img = cv.imread("link.jpg", cv.IMREAD_GRAYSCALE)

h, w = img.shape

mask = np.random.normal(0, 10, (h, w))
mask = mask.reshape(h, w).astype('uint8')
img_mask = cv.add(img, mask)
cv.imshow("Ruido", img_mask)
cv.imwrite("linkRuido.jpg", img_mask)
cv.waitKey(0)
cv.destroyAllWindows()