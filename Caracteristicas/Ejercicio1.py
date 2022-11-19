import cv2 as cv
import numpy as np

img = cv.imread('Carnage.jpeg', cv.IMREAD_GRAYSCALE)

cap = cv.VideoCapture(1)

#SIFT
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)
img = cv.drawKeypoints(img, kp1, None)
#cv.imshow("SIFT KeyPoints", img)
bf = cv.BFMatcher()

while (cap.isOpened()):
    ret, img_ = cap.read(0)
    if ret == True:
        img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(img_, None)
        matches = bf.knnMatch(des1, des2, k=2)
        #Mejores puntos
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        matching_result = cv.drawMatchesKnn(img, kp1, img_, kp2, good, None, flags=2)
        cv.imshow("Matching result", matching_result)
        if cv.waitKey(1) & 0xFF == ord('s'):
              break
    else: break 
        
cv.imshow("Matching result", matching_result)
cv.waitKey(0)
cv.destroyAllWindows()