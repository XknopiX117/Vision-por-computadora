import cv2 as cv
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

fichero = open('Lista.txt', 'w')

cap = cv.VideoCapture(0)

while(True):
    ret, img = cap.read()
    leyenda = 'Colocar tarjeta'
    cv.putText(img, leyenda, (10, 40), cv.FONT_ITALIC, 0.7, (0, 0, 255), 1)
    #cv.imshow('Video', img)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 0)
    d = pytesseract.image_to_data(img_gray, output_type=Output.DICT)
    text = pytesseract.image_to_string(img_gray, lang='spa')
    palabras = text.split()

    nombre = ''
    apellido = ''
    tel = ''
    edad = ''
    dir = ''

    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('video', img_gray)

    if cv.waitKey(1) & 0xFF == ord('c'):
        for w in range(len(palabras)):
            if (palabras[w] == 'Nombre:' or palabras[w] == 'Nombre'):
                if (palabras[w + 2] == 'Apellido:' or palabras[w + 2] == 'Apellido'):
                    nombre += palabras[w + 1]
                else:
                    nombre += palabras[w + 1] + ' ' + palabras[w + 2]

            elif (palabras[w] == 'Apellido:' or palabras[w] == 'Apellido'):
                apellido += palabras[w + 1] + ' ' + palabras[w + 2]

            elif (palabras[w] == 'Teléfono:' or palabras[w] == 'Teléfono' or 
                    palabras[w] == 'Telefono:' or palabras[w] == 'Telefono'):
                tel += palabras[w + 1]

            elif (palabras[w] == 'Edad:' or palabras[w] == 'Edad'):
                edad += palabras[w + 1]

            elif (palabras[w] == 'Dirección:' or palabras[w] == 'Dirección' or 
                    palabras[w] == 'Direccion:' or palabras[w] == 'Direccion'):
                dir_aux = str(palabras[w + 1 : ])
                dir += "".join(dir_aux)
            
        fichero.write('Nombre:\n')
        fichero.write(nombre + '\n')
        fichero.write('Apellido:\n')
        fichero.write(apellido + '\n')
        fichero.write('Teléfono:\n')
        fichero.write(tel + '\n')
        fichero.write('Edad:\n')
        fichero.write(edad + '\n')
        fichero.write('Dirección:\n')
        fichero.write(dir + '\n')

    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.destroyAllWindows()
        fichero.close()
        break