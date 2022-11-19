import cv2 as cv
import numpy as np

img = cv.imread('Carnage.jpeg', cv.IMREAD_GRAYSCALE)

cap = cv.VideoCapture(1)

#ORB Detector
orb = cv.ORB_create(nfeatures=1500)
kp1, des1 = orb.detectAndCompute(img, None)
des1 = np.float32(des1)
img = cv.drawKeypoints(img, kp1, None)
#cv.imshow("SIFT KeyPoints", img)

#FLANN
FIKDTREE = 0
index_params= dict(algorithm = FIKDTREE,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)

while (cap.isOpened()):
    ret, img_ = cap.read(0)
    if ret == True:
        img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(img_, None)
        des2 = np.float32(des2)
        matches = flann.knnMatch(des1, des2, k=2)
        #Mejores puntos
        matchesMask = [[0,0] for i in range(len(matches))]

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
        
        matching_result = cv.drawMatchesKnn(img, kp1, img_, kp2, matches, None, **draw_params)
        cv.imshow("Matching result", matching_result)
        if cv.waitKey(1) & 0xFF == ord('s'):
              break
    else: break 
        
cv.imshow("Matching result", matching_result)
cv.waitKey(0)
cv.destroyAllWindows()