import FlannMatch as fl
import SiftMatch as sm
import FeatureDetector as fd
import HarrisCornerDetector as hcd
import SurfCornerDetector as scd
from matplotlib import pyplot as plt



rows, cols = 2, 2
plt.subplot(rows, cols, 1)
plt.imshow(fl.Flann("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.title("Flann")
plt.subplot(rows, cols, 2)
plt.imshow(sm.Sift("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.title("Sift")
plt.subplot(rows, cols, 3)
plt.imshow(hcd.HarrisCorner("ImageQuery/Fenerbahce.jpg"))
plt.title("HarrisCorner")
plt.subplot(rows, cols, 4)
plt.imshow(fd.ORB("ImageQuery/Fenerbahce.jpg","ImageQuery/fb-gs.jpg"))
plt.title("ORB")

plt.show()
