import numpy as np
import cv2
from moviepy.editor import VideoFileClip


# Define a function to select the region of interest in the image
def region_of_interest(img, vertices):
    """
    Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`.
    The rest of the image is set to black.

    img: The input image.
    vertices: A list of vertices defining the region of interest.
    """
    # Create a blank mask (same size as the input image).
    mask = np.zeros_like(img)

    # Choose the color for the mask depending on the number of color channels in the input image.
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # color image (3 channels)
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  # grayscale image

    # Fill the polygon defined by vertices with the mask color.
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Apply the mask to the image, keeping only the ROI.
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Define a function to draw lines on an image
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draw lines on an image.

    img: The input image on which lines are drawn.
    lines: The line segments to draw, each defined by a pair of points.
    color: The color of the lines (default is red).
    thickness: The thickness of the lines.
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Define a function to apply Hough Transform and detect lines in an image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Applies the Hough Line Transform to detect lines in the image.

    img: The input binary image (result of edge detection).
    rho: Distance resolution of the accumulator in pixels.
    theta: Angle resolution of the accumulator in radians.
    threshold: Accumulator threshold parameter.
    min_line_len: Minimum length of a line (shorter lines are discarded).
    max_line_gap: Maximum allowed gap between points on the same line.
    """
    # Detect lines using the Hough Line Transform.
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # Create an empty image to draw lines on.
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Draw the detected lines on the image.
    draw_lines(line_img, lines)
    return line_img, lines


# Separate lines based on their slope (positive = right lane, negative = left lane)
def separate_left_right_lines(lines):
    """
    Separates the detected lines into left and right lanes based on their slope.

    lines: List of detected lines.
    """
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2:  # Negative slope indicates a left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2:  # Positive slope indicates a right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines


# Calculate the average value of a list
def cal_avg(values):
    """
    Calculates the average of a list of values.

    values: A list of numeric values.
    """
    if values is not None and len(values) > 0:
        return sum(values) / len(values)
    return 0


# Extrapolate line segments to create full lane lines
def extrapolate_lines(lines, upper_border, lower_border):
    """
    Extrapolates the line segments to create full lane lines, considering upper and lower borders.

    lines: List of line segments to be extrapolated.
    upper_border: The y-coordinate of the upper border for the lane lines.
    lower_border: The y-coordinate of the lower border for the lane lines.
    """
    slopes = []
    consts = []

    if lines is not None and len(lines) != 0:
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
    return None


# Combine detected lane lines into a single image
def extrapolated_lane_image(img, lines, roi_upper_border, roi_lower_border):
    """
    Generates the final lane lines image by combining left and right lanes.

    img: The input image.
    lines: List of detected lines.
    roi_upper_border: The y-coordinate of the upper border for the lanes.
    roi_lower_border: The y-coordinate of the lower border for the lanes.
    """
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_left, lines_right = separate_left_right_lines(lines)
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)

    # Draw the lane lines on the lanes image if they are not None.
    if lane_left is not None and lane_right is not None:
        draw_con(lanes_img, [[lane_left], [lane_right]])
    return lanes_img


# Fill in the area between the lanes to highlight the driving path
def draw_con(img, lines):
    """
    Fills in the area between the left and right lanes to highlight the driving path.

    img: The image on which to draw the lane area.
    lines: The left and right lane lines.
    """
    points = []
    for x1, y1, x2, y2 in lines[0]:
        points.append([x1, y1])
        points.append([x2, y2])
    for x1, y1, x2, y2 in lines[1]:
        points.append([x2, y2])
        points.append([x1, y1])

    # Convert the points to an integer array and fill the polygon defined by the lane lines.
    points = np.array([points], dtype='int32')
    cv2.fillPoly(img, points, (0, 255, 0))


# Process each frame of the video to detect lanes
def process_image(image):
    """
    Processes each frame of the video to detect lanes and overlays them on the frame.

    image: The input frame from the video.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Select the intensity of the grayscale image.
    gray_select = cv2.inRange(gray, 150, 255)

    # Define the vertices of the region of interest.
    roi_vertices = np.array([[[100, 540], [900, 540], [525, 330], [440, 330]]])

    # Apply the region of interest mask to the grayscale image.
    gray_select_roi = region_of_interest(gray_select, roi_vertices)

    # Apply Canny edge detection to the masked image.
    img_canny = cv2.Canny(gray_select_roi, 50, 100)

    # Apply Gaussian blur to reduce noise.
    canny_blur = cv2.GaussianBlur(img_canny, (5, 5), 0)

    # Apply Hough Transform to detect lines.
    hough, lines = hough_lines(canny_blur, 1, np.pi / 180, 100, 50, 300)

    # Extrapolate the lane lines and combine them into one image.
    lane_img = extrapolated_lane_image(image, lines, 330, 540)

    # Overlay the lane lines on the original image.
    image_result = cv2.addWeighted(image, 1, lane_img, 0.4, 0.0)
    return image_result


# Initialize video capture and writer
video_cap = cv2.VideoCapture('lane1-straight.mp4')
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))
file_out = 'output_lane_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter(file_out, fourcc, 20.0, (frame_width, frame_height))

# Check if the video file was successfully opened
if not video_cap.isOpened():
    print("Error opening video stream or file")

print("Begin processing video... Wait until 'finished' message!")

# Process the video frame by frame
while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Finished processing video")
        break

    # Process the current frame to detect lanes.
    result = process_image(frame)

    # Write the processed frame to the output video file.
    vid_out.write(result)

    # Display the current frame with detected lanes in a window.
    cv2.imshow('Lane Detection', result)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
vid_out.release()
cv2.destroyAllWindows()

# Notify the user that processing is complete
print(f"Processing complete. The output video is saved as {file_out}.")
cv2.waitKey(0)
