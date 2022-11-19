import cv2 as cv
import math
import mediapipe as mp 

#Carga de los emojis
eA = cv.imread('EmojiAlegre.png')
eB = cv.imread('EmojiBeso.png')
eE = cv.imread('EmojiEnojo.jpg')
eS = cv.imread('EmojiSorprendido.png')

# Realizamos la Video Capture y ponemos 0 por que es el de nuestra camara portatil
capture = cv.VideoCapture(0)

# ancho y alto para nuestra ventana 
capture.set(3,1280)
capture.set(4,720)

# Creamos nuestra Funcion De dibujo 
meshDrawing = mp.solutions.drawing_utils 
configureOfDrawing = meshDrawing.DrawingSpec(thickness = 1, circle_radius= 1)

# Creamos un objecto donde almacenamos la malla facial con las emociones 
mpFaceMesh = mp.solutions.face_mesh #llamamos la fncion de la malla
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

# leemos nuestro Fame

while True :
    resize, frame = capture.read()
    # convercion de color de bgr a rgb
    frameRBG = cv.cvtColor( frame, cv.COLOR_BGR2RGB)

    # observamos los resultados
    result = faceMesh.process(frameRBG)

    # creamos una lista donde almacenamos  los resultados
    px = []
    py = []
    list = []
    r = 5
    t = 3

    if result.multi_face_landmarks: # si detectamos algun rostro
        for rostros in result.multi_face_landmarks :
            # meshDrawing.draw_landmarks(frame, rostros, mpFaceMesh.FACE_CONNECTIONS, configureOfDrawing, configureOfDrawing)
            meshDrawing.draw_landmarks(frame, rostros, mpFaceMesh.FACEMESH_CONTOURS, configureOfDrawing, configureOfDrawing)

            # Extraemos los rostro de los puntos detectados 
            for id, point in enumerate(rostros.landmark):
                high, width, c = frame.shape
                # aqui nos entrega un pixel
                x, y = int(point.x * width), int(point.y * high)
                px.append(x)
                py.append(y)
                list.append([id, x, y])
                if len(list) == 468 :
                    # cejas Derecha
                    x1, y1 = list[65][1:]
                    x2, y2 = list[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    longitud1 = math.hypot(x2 - x1 , y2 - y1)

                    # ceja izquierda 
                    x3, y3 = list[295][1:]
                    x4, y4 = list[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2

                    longitud2 = math.hypot(x4 - x3 , y4 - y3)

                    # boca extremos
                    x5, y5 = list[78][1:]
                    x6, y6 = list[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2

                    longitud3 = math.hypot(x6 - x5 , y6 - y5)

                    # boca apertura 
                    x7, y7 = list[13][1:]
                    x8, y8 = list[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2

                    longitud4 = math.hypot(x8 - x7 , y8 - y7)

                    # clasificacion  
                    if longitud1 < 15 and longitud2 < 15 and longitud3 > 0 and longitud3 < 56 and longitud4 < 2:
                        cv.putText(frame, 'Persona Enojada', (50,40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                        frame[0:63, width-60:] = eE 
                    elif longitud1 > 22 and longitud1 < 29 and longitud2 > 22 and longitud2 < 26 and longitud3 > 60 and longitud4 > 0 and longitud4 < 2:
                        cv.putText(frame, 'Persona Alegre', (50,40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255), 3)
                        frame[0:61, width-60:] = eA    
                    elif longitud1 > 25 and longitud2 > 25 and longitud3 > 45 and longitud3 < 53 and longitud4 > 30:
                        cv.putText(frame, 'Persona Sorprendida', (50,40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3) 
                        frame[0:60, width-60:] = eS
                    elif longitud1 > 25 and longitud1 < 29 and longitud2 > 24 and longitud2 < 29 and longitud3 > 43 and longitud3 < 51 and longitud4 < 5:
                         cv.putText(frame, 'Persona Besando', (50,40), cv.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 3)  
                         frame[0:58, width-60:] = eB  
    
    #print(longitud1, longitud2, longitud3, longitud4)
    cv.imshow("Reconocimiento de Emociones", frame)  
    t = cv.waitKey(1)

    if t == 27 : break
capture.release()
cv.destroyAllWindows() 