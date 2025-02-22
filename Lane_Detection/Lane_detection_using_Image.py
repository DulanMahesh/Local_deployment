import cv2
import numpy as np

#highlevel steps for Lane detection using hough transform in images
#1--->Create a threshold for lane lines -This is done using cv2.inRange on the grayscale image.
#2.--->selecting region of interest-using cv2.fillpoly()
#3.--->Detecting edges using canny edge detector(cv2.canny)
#4.--->Extrapolate the lanes from the lines found -(Lines are detected using the Hough Transform and then extrapolated).
#5.--->composite the result on the original frame (The extrapolated lanes are combined with the original image)

"""
Lane Detection Pipeline Steps
# Create a Threshold for Lane Lines

Purpose: To filter out noise and focus on the relevant lane line colors in the image.
Method:
Convert the image to a different color space (e.g., HSV or LAB) that helps isolate the colors of the lane markings (usually white or yellow).
Apply color thresholding to create a binary mask, where pixels corresponding to lane colors are white (255) and others are black (0).

# Selecting Region of Interest (ROI)

Purpose: To limit the area of the image where lane lines are detected, which reduces computational load and focuses on the road.
Method:
Define a polygonal region that corresponds to the area where the lanes are expected to be (typically the lower part of the image).
Create a mask for this region, applying it to the binary image from the previous step to isolate lane lines in the desired area.

# Detecting Edges using Canny Edge Detector

Purpose: To highlight the edges in the image that correspond to the lane markings.
Method:
Use the Canny Edge Detection algorithm, which detects edges by looking for areas of rapid intensity change.
The process involves applying Gaussian blur to smooth the image, then using gradient operators to find edges and non-maximum suppression to thin them out.

# Fit Lines using Hough Line Transform

Purpose: To detect straight lines in the edge-detected image, which represent the lane markings.
Method:
Apply the Hough Line Transform algorithm, which converts the points in the edge-detected image into lines in Hough space.
This method uses a voting procedure to find the most likely lines based on the detected edges. The parameters can be adjusted to fine-tune the detection (e.g., minimum length of lines, distance between points).

# Extrapolate the Lanes from Lines Found

Purpose: To create a clear lane representation from the detected lines.
Method:
For each detected line, extrapolate (extend) it to the top and bottom of the image to ensure that the lanes are drawn from the bottom (where the vehicle is) to the top of the image.
This may involve calculating the slope and intercept of each line and using those to determine the endpoints for drawing.

# Composite the Result with the Original Frame

Purpose: To visualize the detected lane markings over the original image.
Method:
Create a copy of the original image and draw the extrapolated lane lines onto it using a specified color (usually green or yellow for visibility).
The final output is an image that shows the original scene with the detected lanes highlighted.
"""


# Function to calculate average of a list of values.The average calculation is used in the function extrapolate_lines, where it helps in determining the average slope and y-intercept of the detected lines. This is critical for accurately drawing a single, continuous lane line that represents multiple detected line segments.
def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n


# Function to draw lines on an image using OpenCV's cv2.line.
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Function to  Separate lines into left and right based on the slope.
def separate_left_right_lines(lines):
    """ Separate left and right lines depending on the slope. """
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2:  # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2:  # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines


# Function to  Extrapolate detected lines to span the region of interest.
def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            slope = (y1 - y2) / (x1 - x2)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)

    # Calculate average intersection at lower_border.
    x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)

    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)

    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]

##Main Scrpit --Load and display the input image
if __name__ == "__main__":

    # Reading the image.
    img = cv2.imread('./test_img1.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use global threshold based on grayscale intensity.
    threshold = cv2.inRange(gray, 150, 255)

    # Display images.
    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Region masking: Select vertices according to the input image.
    roi_vertices = np.array([[[100, 540],
                              [900, 540],
                              [515, 320],
                              [450, 320]]])

    # Defining a blank mask.
    mask = np.zeros_like(threshold)

    # Defining a 3 channel or 1 channel color to fill the mask.
    if len(threshold.shape) > 2:
        channel_count = threshold.shape[2]  # 3 or 4 depending on the image.
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon.
    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

    # Constructing the region of interest based on where mask pixels are nonzero.
    roi = cv2.bitwise_and(threshold, mask)

    cv2.imshow('Initial threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Polyfill mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Isolated roi', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform Edge Detection using the Canny algorithm
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(roi, low_threshold, high_threshold)

    # Smooth with a Gaussian blur.
    kernel_size = 3
    canny_blur = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)

    # Display images.
    cv2.imshow('Edge detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Blurred edges', canny_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Hough line detection -Hough transform parameters set according to the input image.purpose is to Detect lines using the Hough Transform and draw them on a blank image.
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_len = 10
    max_line_gap = 20

    lines = cv2.HoughLinesP(canny_blur, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Draw all lines found onto a new image.
    hough = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(hough, lines)

    print("Found {} lines, including: {}".format(len(lines), lines[0]))
    cv2.imshow('Hough', hough)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extrapolate and draw lines - whole purpose is to Separate detected lines into left and right, then extrapolate and draw them.
    # Define bounds of the region of interest.
    roi_upper_border = 340
    roi_lower_border = 540

    # Create a blank array to contain the (colorized) results.
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Use above defined function to identify lists of left-sided and right-sided lines.
    lines_left, lines_right = separate_left_right_lines(lines)

    # Use above defined function to extrapolate the lists of lines into recognized lanes.
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    draw_lines(lanes_img, [[lane_left]], thickness=10)
    draw_lines(lanes_img, [[lane_right]], thickness=10)

    # Display results- purpose of this code snippet is to Combine the original image with the detected and extrapolated lanes for visualization.
    # Following step is optional and only used in the script for display convenience.
    hough1 = cv2.resize(hough, None, fx=0.5, fy=0.5)
    lanes_img1 = cv2.resize(lanes_img, None, fx=0.5, fy=0.5)
    comparison = cv2.hconcat([hough1, lanes_img1])
    cv2.imshow('Before and after extrapolation', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    alpha = 0.8
    beta = 1.0
    gamma = 0.0
    image_annotated = cv2.addWeighted(img, alpha, lanes_img, beta, gamma)

    # Display the results, and save image to file.
    cv2.imshow('Annotated Image', image_annotated)
    cv2.imwrite('./Lane1-image.jpg', image_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
