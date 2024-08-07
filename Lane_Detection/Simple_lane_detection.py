
import cv2
import numpy as np

def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load image
image = cv2.imread('test_img1.jpg')
if image is None:
    raise ValueError("Image not found")

# Step 1: Create a threshold for lane lines

"""
Purpose: To filter out noise and focus on the relevant lane line colors in the image.
Method:
Convert the image to a different color space (e.g., HSV ) that helps isolate the colors of the lane markings (usually white or yellow).
Apply color thresholding to create a binary mask, where pixels corresponding to lane colors are white (255) and others are black (0).
"""
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Step 1.1: Create a mask for white lane lines
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Combine the masks
mask = cv2.bitwise_or(mask_yellow, mask_white)

# Debugging: Show the mask
show_image('Mask', mask)

# Step 2: Selecting Region of Interest
"""
Purpose: To limit the area of the image where lane lines are detected, which reduces computational load and focuses on the road.
Method:
Define a polygonal region that corresponds to the area where the lanes are expected to be (typically the lower part of the image).
Create a mask for this region, applying it to the binary image from the previous step to isolate lane lines in the desired area.
"""

height, width = mask.shape
roi_vertices = np.array([[(0, height), (width, height), (width // 2, int(height * 0.6))]], dtype=np.int32)
mask_roi = np.zeros_like(mask)
cv2.fillPoly(mask_roi, roi_vertices, 255)
masked_image = cv2.bitwise_and(mask, mask_roi)

# Debugging: Show the masked image
show_image('Masked Image', masked_image)

# Step 3: Detecting Edges using Canny Edge Detector
"""
Purpose: To highlight the edges in the image that correspond to the lane markings.
Method:
Use the Canny Edge Detection algorithm, which detects edges by looking for areas of rapid intensity change.
The process involves applying Gaussian blur to smooth the image, then using gradient operators to find edges and non-maximum suppression to thin them out.

"""
edges = cv2.Canny(masked_image, 50, 150)

# Debugging: Show the edges
show_image('Edges', edges)

# Step 4: Fit lines using Hough Line Transform
"""
Purpose: To detect straight lines in the edge-detected image, which represent the lane markings.
Method:
Apply the Hough Line Transform algorithm, which converts the points in the edge-detected image into lines in Hough space.
This method uses a voting procedure to find the most likely lines based on the detected edges. The parameters can be adjusted to fine-tune the detection (e.g., minimum length of lines, distance between points).

"""

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

# Step 5: Extrapolate the lanes from lines found

"""
Purpose: To create a clear lane representation from the detected lines.
Method:
For each detected line, extrapolate (extend) it to the top and bottom of the image to ensure that the lanes are drawn from the bottom (where the vehicle is) to the top of the image.
This may involve calculating the slope and intercept of each line and using those to determine the endpoints for drawing.

"""
line_image = np.zeros_like(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Step 6: Composite the result with the original frame

"""
Purpose: To visualize the detected lane markings over the original image.
Method:
Create a copy of the original image and draw the extrapolated lane lines onto it using a specified color (usually green or yellow for visibility).
The final output is an image that shows the original scene with the detected lanes highlighted.

"""
composite_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

# Show the final composite image
show_image('Lane Detection', composite_image)
