

"""A background subtraction model creates a statistical model of the background based on previous frames.
This can involve averaging pixel values over time or using more complex methods to account for changes in lighting and other variations.
Creating a Foreground Mask:

When a new frame is captured, it is compared to the background model.
Pixels that differ significantly from the background model are marked as foreground (moving objects), 
while pixels that match the background are marked as background. The result is a binary image (foreground mask) where moving objects are highlighted"""


#Erosion (Morphological Operation)
#Purpose of Erosion:

"""Erosion is used to remove small noise from binary images. In the context of motion detection,
it helps in cleaning up the foreground mask by removing small, isolated pixel groups that are likely noise rather than actual moving objects.
How Erosion Works:
Erosion shrinks the white regions (foreground) in a binary image. It removes pixels on the object boundaries.
It works by sliding a structuring element (a small matrix, usually a square) over the image. If the structuring
element fits entirely within the foreground region, the central pixel of the structuring element remains white. If not, it turns black."""

#2. Work Flow
"""Create a statistical model of the background scene using createBackgroundSubtractorKNN()
For each video frame:
Compare the current frame with the background model to create a foreground mask using the apply() method
Apply erosion to the foreground mask to reduce noise using erode()
Find the bounding rectangle that encompasses all the non-zero points in the foreground mask using boundingRect()"""

import cv2
import numpy as np

# Input video file
input_video = 'motion_test.mp4'
video_cap = cv2.VideoCapture(input_video)
if not video_cap.isOpened():
    print('Unable to open: ' + input_video)

# Get video properties
frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Set output video size
size = (frame_w, frame_h)
size_quad = (int(frame_w), int(frame_h))

# Initialize video writer for the output video

#The VideoWriter object video_out_quad is created with the specified filename, codec, frame rate, and frame size.
# It is then used to write processed video frames to the file video_out_quad.mp4.


video_out_quad = cv2.VideoWriter('video_out_quad.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size_quad) # FourCC (Four Character Code) is a 4-byte code used to specify the video codec.
                                                                                                        #cv2.VideoWriter_fourcc is a helper function that converts the given four-character code into an appropriate integer code.
                                                                                                        #  size_quad- This is a tuple specifying the width and height of the video frames. In this case, size_quad is set to (int(frame_w), int(frame_h)), where frame_w and frame_h are the width and height of the input video frames.
# Function to draw a banner with text on a frame
def drawBannerText(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0), font_thickness=2):
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)


# Initialize background subtractor
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

# Kernel size for erosion
ksize = (5, 5)

# Colors for drawing
red = (0, 0, 255)
yellow = (0, 255, 255)

# Main processing loop
while True:
    ret, frame = video_cap.read()

    if frame is None:
        break
    else:
        frame_erode = frame.copy()

    # Stage 1: Motion detection using foreground mask
    fg_mask = bg_sub.apply(frame)
    motion_area = cv2.findNonZero(fg_mask)
    x, y, w, h = cv2.boundingRect(motion_area)

    # Stage 2: Eroded motion detection
    fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
    motion_area_erode = cv2.findNonZero(fg_mask_erode)
    xe, ye, we, he = cv2.boundingRect(motion_area_erode)

    # Draw bounding boxes for detected motion areas
    if motion_area is not None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), red, thickness=6)
    if motion_area_erode is not None:
        cv2.rectangle(frame_erode, (xe, ye), (xe + we, ye + he), red, thickness=6)

    # Convert foreground masks to color for visualization
    frame_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    frame_fg_mask_erode = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)

    # Annotate frames with text
    drawBannerText(frame_fg_mask, 'Foreground Mask')
    drawBannerText(frame_fg_mask_erode, 'Foreground Mask Eroded')

    # Create a composite frame by stacking original and processed frames
    frame_top = np.hstack([frame_fg_mask, frame])
    frame_bot = np.hstack([frame_fg_mask_erode, frame_erode])
    frame_composite = np.vstack([frame_top, frame_bot])

    # Draw a line to separate the top and bottom halves
    fc_h, fc_w, _ = frame_composite.shape
    cv2.line(frame_composite, (0, int(fc_h / 2)), (fc_w, int(fc_h / 2)), yellow, thickness=1, lineType=cv2.LINE_AA)

    # Resize composite frame for compatibility
    frame_composite = cv2.resize(frame_composite, None, fx=0.5, fy=0.5)

    # Write the composite frame to the output video
    video_out_quad.write(frame_composite)

# Release video resources
video_cap.release()
video_out_quad.release()
