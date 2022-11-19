import cv2 as cv, numpy as np, glob as g, matplotlib.pyplot as plt, random

#Copiar imagen
img = []
path = g.glob("C:/Users/jeres/Desktop/Mejoramiento de imagen Basado en estadisticas y Ruido/Imagenes/*")
for file in path:
    f = cv.imread(file, 0)
    img.append(f)

#Histograma original y con ruido
def Histograma(nueva_img):
    ruido = nueva_img
    hist_o = cv.calcHist([img[0]], [0], None, [256], [0, 256])
    hist_n = cv.calcHist([ruido], [0], None, [256], [0, 256])
    plt.figure("Ventana de Histogramas")
    plt.title("Gráfica")
    plt.plot(hist_n, color='blue', label = 'Histograma con ruido')
    plt.plot(hist_o, color = "red", label = 'Histograma original')
    plt.xlabel('Intensidad de iluminación')
    plt.ylabel('Cantidad de píxeles')
    plt.legend()
    plt.show()

cx, cy = img[0].shape

#Imagen original
cv.namedWindow("Original")
cv.imshow("Original", img[0])
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Gaussiano
gauss = np.random.normal(0, 0.4, img[0].size)
gauss = gauss.reshape(cx, cy).astype('uint8')
img_gauss = cv.add(img[0], gauss)
cv.namedWindow('Gaussiano')
cv.imshow('Gaussiano', img_gauss)
#Histograma(img_gauss)
cv.imwrite("RuidoGaussiano.jpg", img_gauss)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Rayleigh
ray = np.random.rayleigh(10, img[0].size)
ray = ray.reshape(cx, cy).astype('uint8')
img_ray = cv.add(img[0], ray)
cv.namedWindow('Rayleigh')
cv.imshow('Rayleigh', img_ray)
#Histograma(img_ray)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Exponencial
ex = np.random.exponential(10, img[0].size)
ex = ex.reshape(cx, cy).astype('uint8')
img_ex = cv.add(img[0], ex)
cv.namedWindow('Exponencial')
cv.imshow('Exponencial', img_ex)
#Histograma(img_ex)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Sal y Pimienta
img_cpy = img[0].copy()
sal = random.randint(5000, 10000)
for s in range(sal):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 255
pimienta = random.randint(5000, 10000)
for p in range(pimienta):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 0
cv.namedWindow('Sal y pimienta')
cv.imshow('Sal y pimienta', img_cpy)
#Histograma(img_ray)
cv.imwrite("RuidoSalPimienta.jpg", img_cpy)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Sal
img_cpy = img[0].copy()
sal = random.randint(5000, 10000)
for s in range(sal):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 255
cv.imshow("Sal", img_cpy)
cv.imwrite("RuidoSal.jpg", img_cpy)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo pimienta
img_cpy = img[0].copy()
pimienta = random.randint(5000, 10000)
for p in range(pimienta):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 0
cv.imshow("Pimienta", img_cpy)
cv.imwrite("RuidoPimienta.jpg", img_cpy)
cv.waitKey(0)
cv.destroyAllWindows()

#Ruido uniforme

#Ruido gauss sal y pimienta
gauss = np.random.normal(0, 0.4, img[0].size)
gauss = gauss.reshape(cx, cy).astype('uint8')
img_gauss = cv.add(img[0], gauss)
img_cpy = img_gauss
sal = random.randint(5000, 10000)
for s in range(sal):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 255
pimienta = random.randint(5000, 10000)
for p in range(pimienta):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 0
cv.namedWindow('Gauss Sal y pimienta')
cv.imshow('Gauss Sal y pimienta', img_cpy)
#Histograma(img_ray)
cv.imwrite("RuidoGaussianoSalPimienta.jpg", img_cpy)
cv.waitKey(0)
cv.destroyAllWindows()

#Ruido uniforme
uf = np.random.uniform(0, 50, img[0].size)
uf = uf.reshape(cx, cy).astype('uint8')
img_uf = cv.add(img[0], uf)
cv.imshow("Uniforme", img_uf)
cv.imwrite("RuidoUniforme.jpg", img_uf)
cv.waitKey(0)
cv.destroyAllWindows()

#Tipo Sal y Pimienta 20%
img_cpy = img[0].copy()
sal = random.randint(1000, 2000)
for s in range(sal):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 255
pimienta = random.randint(1000, 2000)
for p in range(pimienta):
    y_cord = random.randint(0, cy - 1)
    x_cord = random.randint(0, cx - 1)
    img_cpy[y_cord, x_cord] = 0
cv.namedWindow('Sal y pimienta 20%')
cv.imshow('Sal y pimienta 20%', img_cpy)
#Histograma(img_ray)
cv.imwrite("RuidoSalPimienta20.jpg", img_cpy)
cv.waitKey(0)
cv.destroyAllWindows()