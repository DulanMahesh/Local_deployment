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
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)


# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# make a copy of original image
image_with_contours = image.copy()


#displaying bounding box -
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  #cv2.boundingRect(cnt) computes the bounding rectangle for the contour cnt. x and y are the coordinates of the top-left corner of the rectangle. w and h are the width and height of the rectangle, respectively.
    cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2) #cv2.rectangle() draws a rectangle on image_with_contours.(x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner of the rectangle.(0, 255, 0) is the color of the rectangle in BGR format (green in this case).2 is the thickness of the rectangle border

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))# converts the image from BGR (used by OpenCV) to RGB (used by Matplotlib).
plt.title('Image with Bounding Boxes') ;plt.axis('off')
plt.show()


# verticalbounding box - use particularly when the orientation of objects or features in the image is vertically aligned.
image_verticalbounding_box = image.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image_verticalbounding_box, (x, y), (x+w, y+h), (0, 255, 0), 4)
plt.imshow(image_verticalbounding_box[:,:,::-1]);
plt.title('Image with vertical bounding Boxes');plt.axis('off')
plt.show()

#rotated bounding box -The rotated bounding boxes are particularly useful when the objects in the image are not aligned with the horizontal or vertical axes, allowing for more accurate region-of-interest representation.


image_rotated_bounding_box = image.copy()

# Iterate through each contour detected in the image
for cnt in contours:
    # Find the minimum area rectangle that can fit around the contour
    box = cv2.minAreaRect(cnt) #  cv2.minAreaRect(cnt) calculates the minimum area rectangle that can fit around the contour (cnt). This rectangle is rotated, meaning it can have an angle relative to the horizontal axis.

    # Get the four corners of the rectangle as integers
    boxPts = np.int_(cv2.boxPoints(box)) # cv2.boxPoints(box) computes the four corners of the rotated rectangle (box) as floating-point coordinates. np.int_() converts these coordinates to integers, which are required by cv2.drawContours().

    # Draw the contour of the rectangle on the copied image
    cv2.drawContours(image_rotated_bounding_box, [boxPts], -1, (0, 255, 0), 4) #  'cv2.drawContours()' draws contours on image_rotated_bounding_box. [boxPts]: A list containing the points of the contour (in this case, the rotated rectangle).-1: Indicates drawing all contours in the list ([boxPts]).

# Display the image with matplotlib
plt.imshow(image_rotated_bounding_box[:, :, ::-1])  # Convert BGR to RGB for display
plt.title('Image with rotated bounding Boxes')
plt.axis('off')
plt.show()

# fit circles into contours - The purpose is to visualize the minimum enclosing circles around detected contours

# Make a copy of the original image
imageCircle = image.copy()

# Iterate through each contour detected in the image
for cnt in contours:
    # Find the minimum enclosing circle that can fit around the contour
    ((x, y), radius) = cv2.minEnclosingCircle(cnt) # cv2.minEnclosingCircle(cnt) calculates the smallest circle that can completely enclose the contour (cnt). It returns the center coordinates (x, y) of the circle and its radius.

    # Draw the circle on the copied image
    cv2.circle(imageCircle, (int(x), int(y)), int(round(radius)), (0, 0, 255), 2) # (int(x), int(y)): The center of the circle. x and y are converted to integers.

# Display the image with matplotlib
plt.imshow(imageCircle[:, :, ::-1])  # Convert BGR to RGB for display
plt.title('Image with circles fit into contours')  # Set the title of the plot
plt.axis('off')
plt.show()  # Display the plot



#fit elipse into contour

# Make a copy of the original image
imageElipse = image.copy()

# Iterate through each contour detected in the image
for cnt in contours:
    # Fit an ellipse only if the contour has at least 5 points . however if  you want to have contours with less than 5 points you could code it to have a bounding rectangle instead inside the "if" statement.
    if len(cnt) < 5:    # This condition checks if the contour has fewer than 5 points. An ellipse cannot be fitted to fewer than 5 points, so the code skips any contours that do not meet this requirement.
        continue
    ellipse = cv2.fitEllipse(cnt)
    # Draw the ellipse on the copied image
    cv2.ellipse(imageElipse, ellipse, (255, 0, 0), 2)

# Display the image with matplotlib
plt.imshow(imageElipse[:,:,::-1])  # Convert BGR to RGB for display
plt.title('Image with ELIPSE fit into contours')
plt.axis('off')
plt.show()
