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

#apply biliteral filters

# Load image with gaussian noise.
image1 = cv2.imread('mri-skull-20-percent-gaussian-noise.jpg')
image2 = cv2.imread('mri-skull-40-percent-gaussian-noise.jpg')


# diameter of the pixel neighborhood used during filtering.
dia = 20

# Larger the value the distant colours will be mixed together
# to produce areas of semi equal colors.
sigmaColor = 200

# Larger the value more the influence of the farther placed pixels
# as long as their colors are close enough.
sigmaSpace = 100

# Apply bilateralFilter.
dst1 = cv2.bilateralFilter(image1, dia, sigmaColor, sigmaSpace)
dst2 = cv2.bilateralFilter(image2, dia, sigmaColor, sigmaSpace)

plt.figure(figsize = (20, 12))
plt.subplot(221); plt.axis('off'); plt.imshow(image1[:,:,::-1]); plt.title("Image with 20% gaussian noise")
plt.subplot(222); plt.axis('off'); plt.imshow(dst1[:,:,::-1]);   plt.title("Bilateral blur Result")
plt.subplot(223); plt.axis('off'); plt.imshow(image2[:,:,::-1]); plt.title("Image with 40% gaussian noise")
plt.subplot(224); plt.axis('off'); plt.imshow(dst2[:,:,::-1]);   plt.title("Bilateral blur Result")

plt.show()

# application of biliteral filters as skin smoothing filters(as used in social media applications)

# Load images.
img1 = cv2.imread('face-original.jpg')
img2 = cv2.imread('girl-skin.jpg')

# Apply Gaussian filter for comparison.
img1_gaussian = cv2.GaussianBlur(img1, (5,5), cv2.BORDER_DEFAULT)
img2_gaussian = cv2.GaussianBlur(img2, (5,5), cv2.BORDER_DEFAULT)

# Apply bilateralFilter.
img1_bilateral = cv2.bilateralFilter(img1, d = 25, sigmaColor = 90, sigmaSpace = 40)
img2_bilateral = cv2.bilateralFilter(img2, d = 30, sigmaColor = 65, sigmaSpace = 15)

# Display.
plt.figure(figsize = (18, 15))
plt.subplot(321); plt.axis('off'); plt.imshow(img1[:,:,::-1]);           plt.title('Image1')
plt.subplot(322); plt.axis('off'); plt.imshow(img2[:,:,::-1]);           plt.title('Image2')
plt.subplot(323); plt.axis('off'); plt.imshow(img1_gaussian[:,:,::-1]);  plt.title('Gaussian Filter')
plt.subplot(324); plt.axis('off'); plt.imshow(img2_gaussian[:,:,::-1]);  plt.title('Gaussian Filter')
plt.subplot(325); plt.axis('off'); plt.imshow(img1_bilateral[:,:,::-1]); plt.title('Bilateral Filter')
plt.subplot(326); plt.axis('off'); plt.imshow(img2_bilateral[:,:,::-1]); plt.title('Bilateral Filter')

plt.show()