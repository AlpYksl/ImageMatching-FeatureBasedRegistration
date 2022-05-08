import cv2
import numpy as np
import matplotlib.pyplot as plt
import FeatureDetector as fd
# Open the image files.
img1 = cv2.imread("ImageMatch/osymkitapcik.jpg")  # Image to be aligned.
img2= cv2.imread("ImageMatch/Kitap_202006271656344909921.jpg")  # Reference image.

# Convert to grayscale.
img1_color = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_color = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# use ORB to detect keypoints and extract (binary) local
# invariant features
orb = cv2.ORB_create(500)
(kpsA, descsA) = orb.detectAndCompute(img1_color, None)
(kpsB, descsB) = orb.detectAndCompute(img2_color, None)

# match the features
method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
matcher = cv2.DescriptorMatcher_create(method)
matches = matcher.match(descsA, descsB, None)

# Match features between the two images.
# We create a Brute Force matcher with
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Sort matches on the basis of their Hamming distance.
#matches.sort(key=lambda x: x.distance)

# Take the top 90 % matches forward.
matches = matches[:int(len(matches) * 0.9)]
no_of_matches = len(matches)

# Define empty matrices of shape no_of_matches * 2.
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

# loop over the top matches
for (i, m) in enumerate(matches):
	# indicate that the two keypoints in the respective images
	# map to each other
	p1[i] = kpsA[m.queryIdx].pt
	p2[i] = kpsB[m.trainIdx].pt
# Find the homography matrix.
(H, mask) = cv2.findHomography(p1, p2, method=cv2.RANSAC)
# use the homography matrix to align the images
(h, w) = img2_color.shape[:2]
aligned = cv2.warpPerspective(img1_color, H, (w, h))

scale_percent = 60  # percent of original size
width = int(aligned.shape[1] * scale_percent / 100)
height = int(aligned.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
aligned = cv2.resize(aligned, dim, interpolation=cv2.INTER_AREA)
template = cv2.resize(img2_color, dim, interpolation=cv2.INTER_AREA)

# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
stacked = np.hstack([aligned, template])

# our second image alignment visualization will be *overlaying* the
# aligned image on the template, that way we can obtain an idea of
# how good our image alignment is
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
# show the two output image alignment visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
#plt.subplot(2,2,1)
#plt.imshow(fd.ORB("ImageMatch/osymkitapcik.jpg","output.jpg"))
#plt.subplot(2,2,2)
#plt.imshow(fd.ORB("ImageMatch/osymkitapcik.jpg","ImageMatch/Kitap_202006271656344909921.jpg"))
#plt.show()
# Save the output.
cv2.imwrite('output.jpg', aligned)

