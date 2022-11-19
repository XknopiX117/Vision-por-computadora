import cv2 as cv
import numpy as np

def HallarMoneda(radio):
  moneda = 0
  if (10 <= radio <= 11):
      moneda = 1
  elif (12 <= radio <= 13):
      moneda = 2
  elif (14 <= radio <= 16):
      moneda = 5
  elif (17 <= radio <=20):
      moneda = 10
  return moneda

def Cantidad(m1, m2, m5, m10):
  return m1 + 2*m2 + 5*m5 + 10*m10

def Capturar():
  captura = cv.VideoCapture(1)
  while (captura.isOpened()):
    ret, imagen = captura.read()
    h, w = imagen.shape[:2]
    if ret == True:
      cimg = imagen[110:h-110, 170:w-170]
      cv.imshow('Original', cimg)
      img_gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
      cimg2 = np.copy(cimg)

      circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 60, param1=200, param2=20, minRadius=9, maxRadius=20)

      if circles is not None:

        circles = np.uint16(np.around(circles))

        moneda1 = 0
        moneda2 = 0
        moneda5 = 0
        moneda10 = 0

        for i in circles[0,:]:

          cv.circle(cimg2, (i[0],i[1]), i[2], (0,255,0), 2)
          cv.circle(cimg2, (i[0],i[1]), 2, (0,0,255), 3)

          moneda = HallarMoneda(i[2])
          if (moneda == 1):
            moneda1+=1

          elif ((moneda == 2)):
            moneda2+=1

          elif ((moneda == 5)):
            moneda5+=1

          elif ((moneda == 10)):
            moneda10+=1

        if(moneda1 != 0 and moneda2 != 0 and moneda5 != 0 and moneda10 != 0):
          monedas = '$1:'+str(moneda1)+',$2:'+str(moneda2)+',$5:'+str(moneda5)+'$10:'+str(moneda10)

          cantidad = Cantidad(moneda1, moneda2, moneda5, moneda10)
          cantidad = 'Total: '+str(cantidad)

          cv.putText(cimg2, monedas, (0, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
          cv.putText(cimg2, cantidad, (0, 25), cv.FONT_ITALIC, 1, (0, 255, 255), 1)
          cv.imshow('Monedas detectadas', cimg2)

      if cv.waitKey(1) & 0xFF == ord('s'):
        break
    else: break
  captura.release()
  cv.destroyAllWindows()

Capturar()