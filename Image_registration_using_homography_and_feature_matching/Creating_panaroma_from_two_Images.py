import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import matplotlib
plt.rcParams['image.cmap'] = 'gray' # In order to display any image using plt.imshow() or other similar functions will automatically be shown in grayscale

image_file1 = "./scene/scene1.jpg"  # Reference image.
image_file2 = "./scene/scene3.jpg"  # Image to be aligned.

# Read the images.
img1 = cv2.imread(image_file1, cv2.IMREAD_COLOR)
img2 = cv2.imread(image_file2, cv2.IMREAD_COLOR)

# Convert images to grayscale.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#This stage will Compute the Keypoints and Descriptors

# To detect ORB features and compute descriptors.
MAX_FEATURES = 500
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

# To draw the key points
img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color = (0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color = (0,0,255),flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)
plt.figure(figsize = [16, 10])
plt.subplot(121); plt.imshow(img1_keypoints[:,:,::-1]); plt.title("Keypoints for Image-1")
plt.subplot(122); plt.imshow(img2_keypoints[:,:,::-1]); plt.title("Keypoints for Image-2");

plt.show()

#To find Matching Corresponding Points

# To Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Now sort matches by score
matches = sorted(matches, key = lambda x: x.distance, reverse = False)

# Retain only 15%  of the better matches.
GOOD_MATCH_PERCENT = 0.15
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

#Draw the matches
plt.figure(figsize = [15,10])
plt.imshow(img_matches[:,:,::-1])
plt.title("Matches Obtained from the Descriptor Matcher");
plt.show()

#Zoom the image for better understanding and viewing

zoom_x1 = 300; zoom_x2 = 1300
zoom_y1 = 300; zoom_y2 = 700

plt.figure(figsize = [15,10])
img_matches_zoom = img_matches[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
plt.imshow(img_matches_zoom[:,:,::-1])
plt.title("Matches Obtained from the Descriptor Matcher(ZOOMED)");
plt.show()

#Image Alignment using Homography

#Compute the Homography
points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography.
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

#Warp the Image
# Warp time the perspective of the 2nd image using the homography.
img1_h, img1_w, channels = img1.shape
img2_h, img2_w, channels = img2.shape

img2_aligned = cv2.warpPerspective(img2, h, (img2_w + img1_w, img2_h))
plt.figure(figsize = [15,8])
plt.imshow(img2_aligned[:,:,::-1])
plt.title("Second image aligned to first image obtained using homography and warping");
plt.show()

#stich the images by concatanating both images

# Stitch Image-1 with aligned image-2.
stitched_image = np.copy(img2_aligned)
stitched_image[0:img1_h, 0:img1_w] = img1
plt.figure(figsize = [15, 8])
plt.imshow(stitched_image[:,:,::-1])
plt.title("Final Stitched Image");
plt.show()