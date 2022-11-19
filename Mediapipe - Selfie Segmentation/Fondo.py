import cv2 as cv
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Video de entrada
cap = cv.VideoCapture(1)

# Video que se ubica en el fondo
video_name = "CarreteraBosque.mp4"
cap2 = cv.VideoCapture(video_name)

# Color de fondo de escena
BG_COLOR = (219, 203, 255)

with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:

    while True:
        # Lectura del video de entrada
        ret, frame = cap.read()
        if ret == False:
            break

        # Lectura del video que se ubica en el fondo
        ret2, bg_image = cap2.read()
        if ret2 == False:
            cap2 = cv.VideoCapture(video_name)
            ret2, bg_image = cap2.read()

        # Transformar los fotogramas de BGR a RGB y
        # aplicación de MediaPipe Selfie Segmentation
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)

        # Obtener imagen binaria
        _, th = cv.threshold(results.segmentation_mask, 0.5, 255, cv.THRESH_BINARY)

        # Cambio de tipo de dato para poder usarlo con OpenCV
        # e invertir la máscara
        th = th.astype(np.uint8)
        th = cv.medianBlur(th, 13)
        th_inv = cv.bitwise_not(th)


        bg_image = cv.resize(bg_image,(frame.shape[1],frame.shape[0]),interpolation=cv.INTER_CUBIC)

        # Background
        bg = cv.bitwise_and(bg_image, bg_image, mask=th_inv)

        # Foreground
        fg = cv.bitwise_and(frame, frame, mask=th)

        # Background + Foreground
        output_image = cv.add(bg, fg)

        cv.imshow("Nuevo fondo", output_image)
        cv.imshow("Video", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()