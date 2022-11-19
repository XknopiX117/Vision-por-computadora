import cv2 as cv

#Leer en RGB
img1_rgb = cv.imread('Valle de la luna-adicion de ruido gaussiano (1).jpg')
img2_rgb = cv.imread('Lena RGB Sal.tif')

#Cambiar a de RGB a HLS
img1_hls = cv.cvtColor(img1_rgb, cv.COLOR_BGR2HLS)
img2_hls = cv.cvtColor(img2_rgb, cv.COLOR_BGR2HLS)

#Eliminar ruido gaussiano
img1_r1 = cv.GaussianBlur(img1_rgb, (7,7), 1)
img1_r2 = cv.GaussianBlur(img1_hls, (7,7), 1)

#Eliminar ruido sal
img2_r1 = cv.medianBlur(img2_rgb, 5) 
img2_r2 = cv.medianBlur(img2_hls, 5)

#Mostrar imagenes
cv.imshow("Valle RGB", img1_rgb)
cv.imshow("Valle HLS", img1_hls)
cv.imshow("Eliminacion de ruido Valle RGB", img1_r1)
cv.imshow("Eliminacion de ruido Valle HLS", img1_r2)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Lena RGB", img2_rgb)
cv.imshow("Lena HLS", img2_hls)
cv.imshow("Eliminacion de ruido Lena RGB", img2_r1)
cv.imshow("Eliminacion de ruido Lena HLS", img2_r2)
cv.waitKey(0)
cv.destroyAllWindows()