import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# sets the default size of figures (plots) created by Matplotlib. figure will have a width of 20 inches and a height of 10 inches.
#matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
#default colormap used when displaying images with Matplotlibset to gray color
#matplotlib.rcParams['image.cmap'] = 'gray'


# Load an image
image = cv2.imread('shapes.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=[20,10])
# Display the original image
plt.subplot(131);plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.title('Original Image');plt.axis('off')

# Display the grayscale image
plt.subplot(132);plt.imshow(gray_image, cmap='gray');plt.title('Grayscale Image');plt.axis('off')


# Apply thresholding to get a binary image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the binary image
plt.subplot(133);plt.imshow(binary_image, cmap='gray');plt.title('Binary Image');plt.axis('off')
plt.show()

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 1);# (0,0,255) represent B,G,R for the contour color and 2 represent thickness of contour

# Display the image with contours
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.show()

#to explain the center of mass in contours
#following  code iterates over each contour, calculates its moments(Moments are a set of statistical values that give information about the shape of the contour. They are used in image analysis to understand various characteristics of the shape, such as its area, center (centroid), and orientation.), and determines the centroid if the contour has a non-zero area.
#It then draws a small blue circle at the centroid position on the image.
#Finally, the modified image with the centroids marked is displayed using Matplotlib.

for cnt in contours:      # The loop iterates through each contour (cnt) in the contours list. This list is typically obtained from the cv2.findContours() function.
    M = cv2.moments(cnt)  #cv2.moments(cnt) calculates spatial moments up to the third order for the contour cnt. Moments are statistical properties that give information about the shape of an object.M is a dictionary that stores all the calculated moment values. Each key in this dictionary corresponds to a specific moment. For example, M['m00'] represents the area of the contour.
    if M['m00'] != 0:     #M['m00'] represents the zeroth moment (area) of the contour. If the area is zero, it implies that the contour is too small or invalid, and calculating its centroid would result in a division by zero. Hence, the check ensures that only valid contours are processed
        cX = int(M['m10'] / M['m00']) # The centroid (center of mass) coordinates (cX, cY) are calculated using the spatial moments.cX = M['m10'] / M['m00'] calculates the x-coordinate of the centroid.
        cY = int(M['m01'] / M['m00']) # cY = M['m01'] / M['m00'] calculates the y-coordinate of the centroid. int() ensures that the coordinates are integer values, suitable for pixel positions.
        cv2.circle(image_with_contours, (cX, cY), 5, (255, 0, 0), -1) # cv2.circle(image_with_contours, (cX, cY), 5, (255, 0, 0), -1) draws a filled circle (radius 5 pixels) at the centroid position (cX, cY) on the image_with_contours. The color (255, 0, 0) specifies blue in BGR format, and -1 indicates that the circle should be filled.

# Display the image with centers of mass
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)) # This code converts the image from BGR to RGB color space using cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB) because Matplotlib expects RGB images
plt.title('Image with Centers of Mass')
plt.show()


#displaying bounding box
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  #cv2.boundingRect(cnt) computes the bounding rectangle for the contour cnt. x and y are the coordinates of the top-left corner of the rectangle. w and h are the width and height of the rectangle, respectively.
    cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2) #cv2.rectangle() draws a rectangle on image_with_contours.(x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner of the rectangle.(0, 255, 0) is the color of the rectangle in BGR format (green in this case).2 is the thickness of the rectangle border

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Bounding Boxes')
plt.show()



