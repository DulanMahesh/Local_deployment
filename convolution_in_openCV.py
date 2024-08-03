import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the kernel
kernel_size = 5
# Create a 5x5 kernel with all elements equal to 1.
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size ** 2

# Perform convolution
filename = 'kitten.jpg'  # Ensure this file is in our project directory or provide the correct path
image = cv2.imread(filename)

# Check if the image was loaded properly
if image is None:
    print(f"Error: Unable to load image at {filename}")
else:
    dst = cv2.filter2D(image, ddepth=-1, kernel=kernel)

    plt.figure(figsize=[20, 10])
    plt.subplot(121),plt.axis('off'),plt.imshow(image[:, :, ::-1]),plt.title("Original Image")

    plt.subplot(122),plt.axis('off'),plt.imshow(dst[:, :, ::-1]),plt.title("Convolution Result")

    plt.show()

# Apply a box filter - kernel size 5.
box_blur1 = cv2.blur(image, (5,5))

# Apply a box filter - kernel size 11.
box_blur2 = cv2.blur(image, (11,11))

# Display.
plt.figure(figsize = (20, 10))
plt.subplot(121); plt.axis('off'); plt.imshow(box_blur1[:,:,::-1]); plt.title('Box 5x5 kernel')
plt.subplot(122); plt.axis('off'); plt.imshow(box_blur2[:,:,::-1]); plt.title('Box Blur 11x11 kernel');
plt.show()

#Gaussian Blur and affect of Kernal size

# Apply Gaussian blur.
gaussian_blur1 = cv2.GaussianBlur(image, (5,5), 0, 0)# here sigma x and sigma y has set to 0(therefore the blur is calculated from ksize.width and kize.height respectively)
gaussian_blur2 = cv2.GaussianBlur(image, (11,11), 0, 0)

# Display.
plt.figure(figsize = (20, 8))
plt.subplot(121); plt.axis('off'); plt.imshow(gaussian_blur1[:,:,::-1]); plt.title('Gausian Blur 5x5 kernel')
plt.subplot(122); plt.axis('off'); plt.imshow(gaussian_blur2[:,:,::-1]); plt.title('Gausian Blur 11x11 kernel');
plt.show()