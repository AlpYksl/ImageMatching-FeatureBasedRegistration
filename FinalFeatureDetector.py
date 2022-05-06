import glob
import cv2
import matplotlib.pyplot as plot
import os
path = 'ImageQuery'
images  = []
classNames = []
myList = os.listdir(path)
orb = cv2.ORB.create(nfeatures=1000)
print(myList,len(myList))

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findId(img,deslist,threshold =15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal  =-1
    try:
     for des1 in deslist:
         matches = bf.knnMatch(des1,des2,k=2)
         good = []
         for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
         matchList.append(len(good))
    except:
        pass
    if len(matchList) != 0:
        if max(matchList) > threshold:
            finalVal = matchList.index(max(matchList))
    return finalVal
img = cv2.imread("ImageMatch/DaysGone2.jpg")
desList = findDes(images)
image = cv2.imread(path)
   
# Window name in which image is displayed
window_name = 'Image'
  
# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point = (5, 5)
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (220, 220)
  
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
  
while True:
    
    imgOriginal = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    id = findId(img,desList)
    if id != -1:
        cv2.putText(imgOriginal,classNames[id],(50,50)
                    ,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.rectangle(imgOriginal, start_point, end_point, color, thickness)  
       
    cv2.imshow(window_name,imgOriginal)
    cv2.waitKey(1)