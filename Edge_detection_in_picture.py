import cv2
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray'

# Canny Edge Detection
img = cv2.imread('coca-cola-logo.png')

# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img_gray, threshold1 = 180, threshold2 = 200)

plt.figure(figsize = (20,10))
plt.subplot(131); plt.axis("off"); plt.imshow(img[:,:,::-1]); plt.title('Original')
plt.subplot(132); plt.axis("off"); plt.imshow(img_gray);      plt.title('Grayscale')
plt.subplot(133); plt.axis("off"); plt.imshow(edges);         plt.title('Canny Edge Map');
plt.show()
#--------------------------------------------------------------------------------
#to show effects of threshold 2

img = cv2.imread('phone_ipad.jpg')

# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 300)
edges2 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 500)
edges3 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 1000)

plt.figure(figsize = (20,12))
plt.subplot(221); plt.axis("off"); plt.imshow(img_gray);  plt.title('Grayscale')
plt.subplot(222); plt.axis("off"); plt.imshow(edges1);    plt.title('Edges with Threshold2 = 300')
plt.subplot(223); plt.axis("off"); plt.imshow(edges2);    plt.title('Edges with Threshold2 = 500')
plt.subplot(224); plt.axis("off"); plt.imshow(edges3);    plt.title('Edges with Threshold2 = 1000');
plt.show()
#-----------------------------------------------------------------------------------------
#canny edge detection (without blurring)

# Read image.
img1 = cv2.imread('butterfly.jpg')
img2 = cv2.imread('Large_Scaled_Forest_Lizard.jpg')

# Convert to gray scale.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Canny Edge detection without blurring.
original_edges_1 = cv2.Canny(img1_gray, threshold1 = 180, threshold2 = 200)
original_edges_2 = cv2.Canny(img2_gray, threshold1 = 180, threshold2 = 200)

# Display.
plt.figure(figsize = (20,15))
plt.subplot(221); plt.axis('off'); plt.imshow(img1[:,:,::-1]);   plt.title('Original')
plt.subplot(222); plt.axis('off'); plt.imshow(original_edges_1); plt.title('Edges from original-without blur')

plt.subplot(223); plt.axis('off'); plt.imshow(img2[:,:,::-1]);   plt.title('Original')
plt.subplot(224); plt.axis('off'); plt.imshow(original_edges_2); plt.title('Edges from original-without blur');
plt.show()

#------------------------------------------------------------------------
#canny edge with blurring(We apply blurring to smooth out the fine texture and reduce noise in images so that only the predominant edges are detected in the image)

# Apply Gaussian blur with kernel size 7x7.
img1_blur = cv2.GaussianBlur(img1_gray, (7,7), 0)
# Apply Gaussian blur with kernel size 7x7 as the noise is more.
img2_blur = cv2.GaussianBlur(img2_gray, (7,7), 0)

# Display the images with blurring.
plt.figure(figsize = (20, 10))
plt.subplot(231); plt.axis('off'); plt.imshow(img1[:,:,::-1]); plt.title('Original')
plt.subplot(232); plt.axis('off'); plt.imshow(img1_gray);      plt.title('Grayscale')
plt.subplot(233); plt.axis('off'); plt.imshow(img1_blur);      plt.title('Blurred')
plt.subplot(234); plt.axis('off'); plt.imshow(img2[:,:,::-1]); plt.title('Original')
plt.subplot(235); plt.axis('off'); plt.imshow(img2_gray);      plt.title('Grayscale')
plt.subplot(236); plt.axis('off'); plt.imshow(img2_blur);      plt.title('Blurred');
plt.show()

#Perform edge detection using blurred images
blurred_edges_1 = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
blurred_edges_2 = cv2.Canny(img2_blur, threshold1 = 180, threshold2 = 200)

# Display.
plt.figure(figsize = (18,12))
plt.subplot(221); plt.axis('off'); plt.imshow(original_edges_1); plt.title('Edges without blur')
plt.subplot(222); plt.axis('off'); plt.imshow(blurred_edges_1);  plt.title('Edges with blur')
plt.subplot(223); plt.axis('off'); plt.imshow(original_edges_2); plt.title('Edges without blur')
plt.subplot(224); plt.axis('off'); plt.imshow(blurred_edges_2);  plt.title('Edges with blur');
plt.show()

#thresholding Example (effect of Threshold1)
 #Edge detection with a high Threshold1 value.
blurred_edges_tight = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
# Edge detection with a low Threshold1 value.
blurred_edges_open  = cv2.Canny(img1_blur, threshold1 = 50, threshold2 = 200)

plt.figure(figsize = (20,15))
plt.subplot(121); plt.axis('off'); plt.imshow(blurred_edges_tight); plt.title('Threshold1 = 180, Threshold2 = 200')
plt.subplot(122); plt.axis('off'); plt.imshow(blurred_edges_open);  plt.title('Threshold1 = 50, Threshold2 = 200');
plt.show()