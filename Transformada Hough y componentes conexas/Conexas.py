import cv2 as cv
import numpy as np

def HallarMoneda(area):
  moneda = 0
  if (300 <= area <= 320):
      moneda = 1
  elif (430 <= area <= 480):
      moneda = 2
  elif (610 <= area <= 660):
      moneda = 5
  elif (1020 <= area <= 1080):
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
      _, img_ubr = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
      cimg2 = np.copy(cimg)
      cv.imshow('Umbral', img_ubr)

      count, labels, stats, centroids = cv.connectedComponentsWithStats(img_ubr, 8, cv.CV_32S)
      
      moneda1 = 0
      moneda2 = 0
      moneda5 = 0
      moneda10 = 0
      Cant = 0

      for c in range(count):
        x = int(centroids[c, 0])
        y = int(centroids[c, 1])

        area = stats[labels[y, x], cv.CC_STAT_AREA]
        width = stats[labels[y, x], cv.CC_STAT_WIDTH]
        height = stats[labels[y, x], cv.CC_STAT_HEIGHT]
        x_aux = stats[labels[y, x], cv.CC_STAT_LEFT]
        y_aux = stats[labels[y, x], cv.CC_STAT_TOP]
        cw = width//2
        ch = height//2
        r = (cw + ch)//2

        #print(area)
        if (area > 0):
          Cant+=1
          if(area < 76917):
            print(area)
            cv.circle(cimg2, (x_aux+cw,y_aux+ch), r, (0,255,0), 2)
            cv.circle(cimg2, (x_aux+cw,y_aux+ch), 2, (0,0,255), 3)

            moneda = HallarMoneda(area)
            if (moneda == 1):
              moneda1 += 1
            elif (moneda == 2):
              moneda2+=2
            elif (moneda == 5):
              moneda5+=1
            elif (moneda == 10):
              moneda10+=1

      #if(moneda1 != 0 and moneda2 != 0 and moneda5 != 0):
      monedas = '$1:'+str(moneda1)+',$2:'+str(moneda2)+',$5:'+str(moneda5)+',$10:'+str(moneda10)
      total = 'Total: '+str(Cantidad(moneda1, moneda2, moneda5, moneda10))
      cv.putText(cimg2, monedas, (0, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
      cv.putText(cimg2, total, (0, 25), cv.FONT_ITALIC, 1, (0, 255, 255), 1)
      cv.imshow('Monedas detectadas', cimg2)
      if cv.waitKey(1) & 0xFF == ord('s'):
        break
    else: break
  captura.release()
  cv.destroyAllWindows()

Capturar()