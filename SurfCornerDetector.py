

#Speeded-up robust features

import numpy as np
import cv2 as cv

def Surf(img):
    # Read the image
    img = cv.imread(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the features (i.e. keypoints) and feature descriptors in the image
    surf = cv.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img, None)

    # Draw circles to indicate the location of features and the feature's orientation
    img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save the image
    cv.imwrite('surf_with_features_chessboard.jpg', img)
    return img