import cv2
import numpy as np

# Source video file
source = 'intruder_2.mp4'

# Capture video from the source file
video_cap_2 = cv2.VideoCapture(source)
if not video_cap_2.isOpened():
    print('Unable to open: ' + source)

# Get video frame width, height, and frames per second (fps)
frame_w = int(video_cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap_2.get(cv2.CAP_PROP_FPS))

# Define size and frame area
size = (frame_w, frame_h)
frame_area = frame_w * frame_h

# Output video file for alert
video_out_alert_file_2 = 'video_out_alert_2.mp4'
video_out_alert_2 = cv2.VideoWriter(video_out_alert_file_2, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Function to draw a banner with text on a frame
def drawBannerText(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0), font_thickness=2):
    # Draw a black filled banner across the top of the image frame.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    # Draw text on the banner.
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Initialize background subtractor using the K-Nearest Neighbors method
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

# Kernel size for erosion
ksize = (5, 5)

# Number of contours to use for rendering a bounding rectangle
max_contours = 3

# Frame counter
frame_count = 0

# Minimum fraction of frame required for maximum contour
min_contour_area_thresh = 0.01

# Colors for drawing
yellow = (0, 255, 255)
red = (0, 0, 255)

# Process video frames
while True:
    # Read a frame from the video
    ret, frame = video_cap_2.read()
    frame_count += 1
    if frame is None:
        break

    # Stage 1: Create a foreground mask for the current frame
    fg_mask = bg_sub.apply(frame)

    # Stage 2: Apply erosion to the foreground mask
    fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))

    # Stage 3: Find contours in the eroded foreground mask
    contours_erode, hierarchy = cv2.findContours(fg_mask_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_erode) > 0:
        # Sort contours based on area
        contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)
        """here sorted()gives the erorded contours from small contour to larger. but since we need the large contour first, we enable the 'reverse =True' """

        # Contour area of the largest contour
        contour_area_max = cv2.contourArea(contours_sorted[0]) # cv2.contourArea computes the area of a contour. Here, contours_sorted[0] refers to the largest contour, which is the first element in the sorted list of contours (contours_sorted). This list is sorted in descending order based on the contour area.

        # Compute the fraction of total frame area occupied by the largest contour
        contour_frac = contour_area_max / frame_area
        """here calculates the fraction of the total frame area that is occupied by the largest contour. 
        The total frame area (frame_area) is computed as the product of the frame's width and height (frame_w * frame_h). Dividing the area of the largest contour by the total frame area gives  the proportion 
        of the frame occupied by this contour."""


        # Check if the contour fraction is greater than the minimum threshold
        if contour_frac > min_contour_area_thresh: # To filter out small, insignificant contours that might be false positives we checks whether the (contour_frac) is greater than a predefined threshold (min_contour_area_thresh).
            # Compute bounding rectangle for the top N largest contours
            for idx in range(min(max_contours, len(contours_sorted))): #  This loop iterates through the top N largest contours, where N is defined by max_contours. The function min(max_contours, len(contours_sorted)) ensures that we do not attempt to access more contours than are available in the contours_sorted list.
                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx]) # cv2.boundingRect calculates the bounding rectangle for a given contour. This function returns the coordinates of the top-left corner of the rectangle (xc, yc) and the width wc and height hc of the rectangle.
                if idx == 0:    # For the first contour (that is, idx == 0), the bounding rectangle coordinates are initialized. x1 and y1 represent the top-left corner, and x2 and y2 represent the bottom-right corner of the rectangle.
                    x1 = xc
                    y1 = yc
                    x2 = xc + wc
                    y2 = yc + hc
                else:         #For each of these subsequent contours, the coordinates of the overall bounding rectangle are adjusted to make sure it covers not just the current contour, but also all previous contours that have been processed. This is done by comparing the current contour's bounding rectangle with the existing bounding rectangle and adjusting the boundaries if necessary.
                    x1 = min(x1, xc) # Adjust x1 if the new contour's xc is less than current x1
                    y1 = min(y1, yc) # Adjust y1 if the new contour's yc is less than current y1
                    x2 = max(x2, xc + wc) # Adjust x2 if the new contour's xc + wc is greater than current x2
                    y2 = max(y2, yc + hc) # Adjust y2 if the new contour's yc + hc is greater than current y2

            # Draw bounding rectangle for the top N contours on the output frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, thickness=2)

            # Draw banner text for intrusion alert
            drawBannerText(frame, 'Intrusion Alert', text_color=red)

            # Write alert video to the file system
            video_out_alert_2.write(frame)

# Release video resources
video_cap_2.release()
video_out_alert_2.release()



