import cv2
import numpy as np

img1 = cv2.imread("ImageQuery/Fenerbahce.jpg")
img2 = cv2.imread("ImageQuery/fb-gs.jpg")

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB.create(nfeatures=1000)

kp1,des1 = orb.detectAndCompute(gray_img1,None)
kp2,des2 = orb.detectAndCompute(gray_img2,None)
print(des1[0])


#imgKp1= cv2.drawKeypoints(img1,kp1,None)
#imgKp2 = cv2.drawKeypoints(img2,kp2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

good = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

#cv2.imshow('Kp1',imgKp1)
#cv2.imshow('Kp2',imgKp2)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.waitKey(0)