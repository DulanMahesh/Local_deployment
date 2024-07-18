import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Quiz5_3_2.png')

# Convert to HSV color space for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for thresholding
# Assuming circles are bright, we threshold on the V channel
lower_val = np.array([0, 0, 100])
upper_val = np.array([255, 255, 255])

# Create a binary mask
mask = cv2.inRange(hsv, lower_val, upper_val)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw all the contours on a copy of the original image
output_image = image.copy()
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
