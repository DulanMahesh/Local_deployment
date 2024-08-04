import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an example image with multiple shapes
image = np.zeros((400, 400), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (350, 350), 255, -1)  # Outer square
cv2.rectangle(image, (150, 150), (250, 250), 0, -1)  # Inner square
cv2.circle(image, (200, 200), 50, 255, -1)           # Circle inside the inner square

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and calculate centroids
image_with_contours = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        cv2.circle(image_with_contours, (cX, cY), 5, (255, 0, 0), -1)  # Draw centroid

# Display the image with contours and centroids
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Centers of Mass')
plt.axis('off')
plt.show()
