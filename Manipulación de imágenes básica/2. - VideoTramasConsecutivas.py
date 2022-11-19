import cv2, numpy as np

#Inicio de captura
captura = cv2.VideoCapture(1)
_, img = captura.read()
anterior = img.copy()

#Captura del video
while (captura.isOpened()):
  ret, imagen = captura.read()
  if ret == True:
    cv2.imshow('video', imagen)
    #Diferencia absoluta
    diferencia = cv2.absdiff(imagen, anterior)
    _, diferencia = cv2.threshold(diferencia, 32, 0, cv2.THRESH_TOZERO)
    _, diferencia = cv2.threshold(diferencia, 0, 255, cv2.THRESH_BINARY)
		
    diferencia = cv2.medianBlur(diferencia, 5)
		
    cv2.imshow('Actual', imagen)
    cv2.imshow('Anterior', diferencia)
    anterior = imagen.copy()
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
  else: break

#Cerrar captura
captura.release()
cv2.destroyAllWindows()