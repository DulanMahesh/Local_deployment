import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def region_of_interest(img, vertices):
    """Select the region of interest (ROI) from a defined list of vertices."""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
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

def cal_avg(values):
    """Calculate average value."""
    if values is not None and len(values) > 0:
        return sum(values) / len(values)
    return 0

def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
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

def extrapolated_lane_image(img, lines, roi_upper_border, roi_lower_border):
    """Main function called to get the final lane lines."""
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_left, lines_right = separate_left_right_lines(lines)
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    if lane_left is not None and lane_right is not None:
        draw_con(lanes_img, [[lane_left], [lane_right]])
    return lanes_img

def draw_con(img, lines):
    """Fill in lane area."""
    points = []
    for x1, y1, x2, y2 in lines[0]:
        points.append([x1, y1])
        points.append([x2, y2])
    for x1, y1, x2, y2 in lines[1]:
        points.append([x2, y2])
        points.append([x1, y1])
    points = np.array([points], dtype='int32')
    cv2.fillPoly(img, points, (0, 255, 0))

def process_image(image):
    """Process each frame of the video to detect lanes."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_select = cv2.inRange(gray, 150, 255)
    roi_vertices = np.array([[[100, 540], [900, 540], [525, 330], [440, 330]]])
    gray_select_roi = region_of_interest(gray_select, roi_vertices)
    img_canny = cv2.Canny(gray_select_roi, 50, 100)
    canny_blur = cv2.GaussianBlur(img_canny, (5, 5), 0)
    hough, lines = hough_lines(canny_blur, 1, np.pi / 180, 100, 50, 300)
    lane_img = extrapolated_lane_image(image, lines, 330, 540)
    image_result = cv2.addWeighted(image, 1, lane_img, 0.4, 0.0)
    return image_result

# Initialize video capture and writer
video_cap = cv2.VideoCapture('lane1-straight.mp4')
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))
file_out = 'output_lane_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter(file_out, fourcc, 20.0, (frame_width, frame_height))

if not video_cap.isOpened():
    print("Error opening video stream or file")

print("Begin processing video... Wait until 'finished' message!")
while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Finished processing video")
        break
    result = process_image(frame)
    vid_out.write(result)
    cv2.imshow('Lane Detection', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_cap.release()
vid_out.release()
cv2.destroyAllWindows()

# Display the saved video file using the default video player
print(f"Processing complete. The output video is saved as {file_out}.")
cv2.waitKey(0)
