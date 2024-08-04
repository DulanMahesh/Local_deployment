import cv2
import numpy as np


#  Canny Edge Detection (simple example with no texture or noise).
img = cv2.imread('coca-cola-logo.png')
# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, threshold1 = 180, threshold2 = 200)

# Display.
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)
cv2.imshow('Canny Edge Map', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Effect of threshold 2.
img = cv2.imread('phone_ipad.jpg')
# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 300)
edges2 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 500)
edges3 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 1000)

# Display.
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 300', edges1)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 500', edges1)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 1000', edges1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny edge detection with and without blurring.
# Read image.
img1 = cv2.imread('butterfly.jpg')
img2 = cv2.imread('Large_Scaled_Forest_Lizard.jpg')

# Resize for display convenience, not mendatory.
img1 = cv2.resize(img1, None, fx = 0.6, fy = 0.6)
img2 = cv2.resize(img2, None, fx = 0.6, fy = 0.6)
# Convert to gray scale.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Canny Edge detection without blurring.
original_edges_1 = cv2.Canny(img1_gray, threshold1 = 180, threshold2 = 200)
original_edges_2 = cv2.Canny(img2_gray, threshold1 = 180, threshold2 = 200)

# Apply Gaussian blur with kernel size 7x7.
img1_blur = cv2.GaussianBlur(img1_gray, (7,7), 0)
# Apply Gaussian blur with kernel size 7x7 as the noise is more.
img2_blur = cv2.GaussianBlur(img2_gray, (7,7), 0)

blurred_edges_1 = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
blurred_edges_2 = cv2.Canny(img2_blur, threshold1 = 180, threshold2 = 200)

compare1 = cv2.hconcat([img1_gray, original_edges_1, blurred_edges_1])
compare2 = cv2.hconcat([img2_gray, original_edges_2, blurred_edges_2])

# Display.
cv2.imshow('Original Gray Scale :: Canny Edge without Blurring :: Canny Edge with Blurring', compare1)
cv2.waitKey(0)
cv2.imshow('Original Gray Scale :: Canny Edge without Blurring :: Canny Edge with Blurring', compare2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hysteresis Thresholding Example (effect of Threshold1).
# Edge detection with a high Threshold1 value.
blurred_edges_tight = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
# Edge detection with a low Threshold1 value.
blurred_edges_open  = cv2.Canny(img1_blur, threshold1 = 50, threshold2 = 200)

# Display.
cv2.imshow('Threshold1 = 180, Threshold2 = 200', blurred_edges_tight)
cv2.waitKey(0)
cv2.imshow('Threshold1 = 50, Threshold2 = 200', blurred_edges_open)
cv2.waitKey(0)
cv2.destroyAllWindows()