import cv2 as cv 
import numpy as np

img = cv.imread("linkRuido.jpg", cv.IMREAD_GRAYSCALE)

'''Degradacion'''
def Degradar(image):

    h, w = image.shape

    img_f = np.float32(image)
    img_fft = np.fft.fft2(img_f)
    img_sfft = np.fft.fftshift(img_fft)

    H = np.copy(img_sfft)
    cy, cx = h//2, w//2
    a, b, T, aux = 0.15, 0.2, 1, 0.0
    for y in range(h):
        for x in range(w):
            aux = np.pi*((y-cy)*a+(x-cx)*b)
            if aux == 0:
                aux = 0.001
            H[y, x] = (T/aux)*np.sin(aux)*np.e**(-1j*aux)

    G = img_sfft*H
    img_isfft = np.fft.ifftshift(G)
    img_ifft = np.fft.ifft2(img_isfft)
    img_deg = np.log(np.absolute(img_ifft))
    img_deg = (img_deg-np.amin(img_deg))/(np.amax(img_deg)-np.amin(img_deg))
    return img_deg, H, G

'''Filtro Inverso'''
def FiltroInverso(h, g):
    F = g/h
    img_isfft = np.fft.ifftshift(F)
    img_ifft = np.fft.ifft2(img_isfft)
    img_rec = np.log(np.absolute(img_ifft))
    img_rec = (img_rec-np.amin(img_rec))/(np.amax(img_rec)-np.amin(img_rec))
    return img_rec

'''Filtro Weiner'''
def Weiner(image, img_deg, H, G):
    img_f = np.float32(image)
    img_fft = np.fft.fft2(img_f)
    img_sfft = np.fft.fftshift(img_fft)
    Sf = np.linalg.norm(img_sfft)**2

    img_f = np.float32(img_deg)
    img_fft = np.fft.fft2(img_f)
    img_sfft = np.fft.fftshift(img_fft)
    Sn = np.linalg.norm(img_sfft)**2

    F = (np.linalg.norm(H)**2)/((np.linalg.norm(H)**2)+(Sn/Sf))*G/H
    img_isfft = np.fft.ifftshift(F)
    img_ifft = np.fft.ifft2(img_isfft)
    img_rec = np.log(np.absolute(img_ifft))
    img_rec = (img_rec-np.amin(img_rec))/(np.amax(img_rec)-np.amin(img_rec))
    return img_rec

'''Filtro Limitacion Radial'''
def LimitacionRadial(img_deg, D, G, H):
    kernel = np.copy(img_deg)
    h, w = img_deg.shape
    cy, cx = h//2, w//2
    for i in range(h):
        for j in range(w):
            dis = np.sqrt(((cy-i)**2)+((cx-j)**2))
            if(dis < D):
                kernel[i, j] = 1
            else:
                kernel[i, j] = 0
    F = kernel*G
    F = F/H
    img_isfft = np.fft.ifftshift(F)
    img_ifft = np.fft.ifft2(img_isfft)
    img_rec = np.log(np.abs(img_ifft))
    img_rec = (img_rec-np.amin(img_rec))/(np.amax(img_rec)-np.amin(img_rec))
    return img_rec

'''Mostrar imagenes'''
cv.imshow("Original", img)
img_deg, H, G = Degradar(img)
img_inverso = FiltroInverso(H, G)
cv.imshow("Degradacion", img_deg)
cv.imshow("Filtro Inverso", img_inverso)
img_weiner = Weiner(img, img_deg, H, G)
cv.imshow("Weiner", img_weiner)
img_lim = LimitacionRadial(img_deg, 80, G, H)
cv.imshow("Limitacion Radial (pasa baja ideal)", img_lim)
cv.waitKey(0)
cv.destroyAllWindows()