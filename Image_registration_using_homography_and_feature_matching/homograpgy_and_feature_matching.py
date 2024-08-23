# Import Libraries.
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.Read and Display Images of Scanned Form and Template

img_form = 'form.jpg'
img_form = cv2.imread(img_form, cv2.IMREAD_COLOR)
img_scan = "scanned-form.jpg"
img_scan = cv2.imread(img_scan, cv2.IMREAD_COLOR)

# Display the images.
plt.figure(figsize = [20, 10])
plt.subplot(121); plt.axis('on'); plt.imshow(img_form[:, :, ::-1]); plt.title("Original Form")
plt.subplot(122); plt.axis('off'); plt.imshow(img_scan[:, :, ::-1]); plt.title("Scanned Form");
plt.show()

#1.2.Find Keypoints in Both Images
# Convert images to grayscale.
img_form_gray = cv2.cvtColor(img_form, cv2.COLOR_BGR2GRAY)
img_scan_gray = cv2.cvtColor(img_scan, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors.
orb = cv2.ORB_create(nfeatures=600)
keypoints1, descriptors1 = orb.detectAndCompute(img_form_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img_scan_gray, None)

# Draw the keypoints in both images.
img_form_keypoints = cv2.drawKeypoints(img_form_gray, keypoints1, None,
                                       color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_scan_keypoints = cv2.drawKeypoints(img_scan_gray, keypoints2, None,
                                       color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the images with the keypoints.
plt.figure(figsize=[20, 10])
plt.subplot(121);plt.axis('off');plt.imshow(img_form_keypoints);plt.title("Original Form")
plt.subplot(122);plt.axis('off');plt.imshow(img_scan_keypoints);plt.title("Scanned Form");
plt.show()

#2.Match Keypoints in the Two Images
# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score.
matches = sorted(matches, key=lambda x: x.distance, reverse=False)


numGoodMatches = int(len(matches) * 0.10) # here we retain only the top 10% of matches.
matches = matches[:numGoodMatches]

# Draw top matches.
im_matches = cv2.drawMatches(img_form_gray, keypoints1, img_scan_gray, keypoints2, matches, None)

plt.figure(figsize = [40,10])
plt.imshow(im_matches); plt.axis('off'); plt.title("Original Form");
plt.show()

#3.find the Homography
# Extract the location of good matches.
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)


#4.warp the perspective of original image
# Use the homography to warp the scanned image.
height, width, channels = img_form.shape
img_scan_reg = cv2.warpPerspective(img_scan, h, (width, height))

# Display the final results.
plt.figure(figsize = [20,10])
plt.subplot(121); plt.imshow(img_form[:, :, ::-1]);     plt.axis('off'); plt.title("Original Form")
plt.subplot(122); plt.imshow(img_scan_reg[:, :, ::-1]); plt.axis('off'); plt.title("Scanned Form");
plt.show()