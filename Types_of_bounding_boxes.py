import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# Load an image
image = cv2.imread('shapes.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=[20,10])

# Apply thresholding to get a binary image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)


# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# make a copy of original image
image_with_contours = image.copy()


#displaying bounding box
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  #cv2.boundingRect(cnt) computes the bounding rectangle for the contour cnt. x and y are the coordinates of the top-left corner of the rectangle. w and h are the width and height of the rectangle, respectively.
    cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2) #cv2.rectangle() draws a rectangle on image_with_contours.(x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner of the rectangle.(0, 255, 0) is the color of the rectangle in BGR format (green in this case).2 is the thickness of the rectangle border

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))# converts the image from BGR (used by OpenCV) to RGB (used by Matplotlib).
plt.title('Image with Bounding Boxes')
plt.show()



