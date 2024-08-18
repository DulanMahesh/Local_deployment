# Import Libraries.
import cv2
import matplotlib.pyplot as plt

# Load an images with Salt and pepper noise.
img1 = cv2.imread('mona_lisa.jpg')
img2 = cv2.imread('ice-flakes-microscopy-salt-and-pepper-noise.jpg')

# Apply median filter.
img1_median = cv2.medianBlur(img1, 9)
img2_median = cv2.medianBlur(img2, 3)

# Apply Gaussian filter for comparison.
img1_gaussian = cv2.GaussianBlur(img1, (5, 5), cv2.BORDER_DEFAULT)
img2_gaussian = cv2.GaussianBlur(img2, (5, 5), cv2.BORDER_DEFAULT)

plt.figure(figsize = (20, 8))
plt.subplot(131); plt.axis('off'); plt.imshow(img1[:,:,::-1]);          plt.title('Original Image with Salt & Pepper Noise')
plt.subplot(132); plt.axis('off'); plt.imshow(img1_gaussian[:,:,::-1]); plt.title('Gaussian filter applied')
plt.subplot(133); plt.axis('off'); plt.imshow(img1_median[:,:,::-1]);   plt.title('Median filter applied')
plt.figure(figsize = (20, 10))
plt.subplot(131); plt.axis('off'); plt.imshow(img2[:,:,::-1]);          plt.title('Original Image with Salt & Pepper Noise')
plt.subplot(132); plt.axis('off'); plt.imshow(img2_gaussian[:,:,::-1]); plt.title('Gaussian filter applied')
plt.subplot(133); plt.axis('off'); plt.imshow(img2_median[:,:,::-1]);   plt.title('Median filter applied')

plt.show()