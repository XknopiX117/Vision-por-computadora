import cv2 as cv

img1 = cv.imread('Farolas-LED.jpg')
img2 = cv.imread('Rocas.tif')

img1 = cv.resize(img1, (0,0), fx = 0.9, fy = 0.9, interpolation=cv.INTER_AREA)
img2 = cv.resize(img2, (0,0), fx = 0.8, fy = 0.8, interpolation=cv.INTER_AREA)

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img1_hls = cv.cvtColor(img1, cv.COLOR_BGR2HLS)
img2_hls = cv.cvtColor(img2, cv.COLOR_BGR2HLS)

img1_hist_bgr = img1
img1_hist_bgr[:, :, 0] = cv.equalizeHist(img1[:,:,0])
img1_hist_bgr[:, :, 1] = cv.equalizeHist(img1[:,:,1])
img1_hist_bgr[:, :, 2] = cv.equalizeHist(img1[:,:,2])

img1_hist_hls = img1_hls
img1_hist_hls[:, :, 2] = cv.equalizeHist(img1_hls[:,:,2])

img2_hist_bgr = img2
img2_hist_bgr[:, :, 0] = cv.equalizeHist(img2[:,:,0])
img2_hist_bgr[:, :, 1] = cv.equalizeHist(img2[:,:,1])
img2_hist_bgr[:, :, 2] = cv.equalizeHist(img2[:,:,2])

img2_hist_hls = img2_hls
img2_hist_hls[:, :, 0] = cv.equalizeHist(img2_hls[:,:,0])

cv.imshow('Original 1', img1)
cv.imshow('RGB 1', img1_rgb)
cv.imshow('HLS 1', img1_hls)
cv.imshow('Histograma ecualizado RGB 1', img1_hist_bgr)
cv.imshow('Histograma ecualizado HLS 1', cv.cvtColor(img1_hist_hls, cv.COLOR_HLS2BGR))
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('Original 2', img2)
cv.imshow('HLS 2', img2_hls)
cv.imshow('RGB 2', img2_rgb)
cv.imshow('Histograma ecualizado RGB 2', img2_hist_bgr)
cv.imshow('Histograma ecualizado HLS 2', cv.cvtColor(img2_hist_hls, cv.COLOR_HLS2BGR))
cv.waitKey(0)
cv.destroyAllWindows()