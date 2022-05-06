import FlannMatch as fl
import SiftMatch as sm
import FeatureDetector as fd
import  HarrisCornerDetector as hcd
from matplotlib import pyplot as plt

#plt.subplot(1,2,1);plt.imshow(fl.Flann("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"));plt.title("Flann")
#plt.subplot(1,2,2);plt.imshow(sm.Sift("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg")) ;plt.title("Sift")
#plt.subplot(2,2,1);plt.imshow(hcd.HarrisCorner("ImageQuery/Fenerbahce.jpg")) ;plt.title("Harris");
#plt.subplot(2,2,2);plt.imshow(fd.ORB("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"));plt.title("ORB");
#plt.show()
#plt.imshow(sfd.Surf("ImageQuery/Fenerbahce.jpg")); plt.show()



# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(fl.Flann("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.axis('off')
plt.title("First")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(sm.Sift("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.axis('off')
plt.title("Second")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(hcd.HarrisCorner("ImageQuery/Fenerbahce.jpg"))
plt.axis('off')
plt.title("Third")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(fd.ORB("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.axis('off')
plt.title("Fourth")