import cv2 as cv, numpy as np, glob as g, matplotlib.pyplot as plt

#Copiar imagen original y a color
img_copia = []
img = []
path = g.glob("C:/Users/jeres/Desktop/Mejoramiento de imagen Basado en estadisticas y Ruido/Imagenes/*")
for file in path:
    f = cv.imread(file)
    c = cv.imread(file, 0)
    img.append(f)
    img_copia.append(c)

cx, cy = img[0].shape[0], img[0].shape[1]

#Original
cv.namedWindow("Foto original")
cv.imshow("Foto original", img[0])
cv.waitKey(0)
cv.destroyAllWindows()

#Identidad
matriz = np.eye(cy, cx)
id = np.dot(matriz, img_copia[0])
plt.imshow(id, cmap="gray")
plt.show()

#Escalamiento
scl = cv.resize(img[0], (0,0), fx= 0.5, fy=1.25, interpolation = cv.INTER_CUBIC)
cv.namedWindow("Escalamiento")
cv.imshow("Escalamiento", scl)
cv.waitKey(0)
cv.destroyAllWindows()

#Rotacion
matriz = cv.getRotationMatrix2D((cx//2, cy//2), 45, 1)
rot = cv.warpAffine(img[0], matriz, (cx, cy))
cv.namedWindow("Rotacion a 45 grados")
cv.imshow("Rotacion a 45 grados", rot)
cv.waitKey(0)
cv.destroyAllWindows()

#Traslacion
matriz = np.float32([[1, 0, 10],[0, 1, 100]])
tras = cv.warpAffine(img[0], matriz, (cx, cy))
cv.namedWindow("Traslacion")
cv.imshow("Traslacion", tras)
cv.waitKey(0)
cv.destroyAllWindows()

#Trapezoidal
n = np.float32([[56, 65], [368, 52], [30, 387], [389, 390]])
m = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
matriz = cv.getPerspectiveTransform(n, m)
trpz = cv.warpPerspective(img[0], matriz, (cx, cy))
cv.namedWindow("Trapezoidal")
cv.imshow('Trapezoidal', trpz)
cv.waitKey(0)
cv.destroyAllWindows()

#Shear
n = np.float32([[0, 0], [cx, 0], [0, cy]])
m = np.float32([[20, 20], [cx - 120, cy / 4], [cx / 4, cy - 120]])
matriz = cv.getAffineTransform(n, m)
shr = cv.warpAffine(img[0], matriz, (cx, cy))
cv.namedWindow("Shear")
cv.imshow("Shear", shr)
cv.waitKey(0)
cv.destroyAllWindows()