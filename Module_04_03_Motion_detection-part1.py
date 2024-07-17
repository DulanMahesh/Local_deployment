

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
    """banner_height: This calculates the height of the banner based on the percentage of the frame height. 
    frame.shape[0] gives the height of the frame, and multiplying it by banner_height_percent gives the height of the banner in pixels. 
    This value is then converted to an integer"""

    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)
    """This draws a filled rectangle (the banner) on the top of the frame. (0, 0) is the top-left corner of the rectangle.
    (frame.shape[1], banner_height) is the bottom-right corner of the rectangle. frame.shape[1] gives the width of the frame.
    (0, 0, 0) specifies the color of the rectangle, which is black. thickness=-1 indicates that the rectangle should be filled"""

#calculate text location

    left_offset = 20 #This sets a horizontal offset (20 pixels) from the left edge of the frame for the text.
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    """This calculates the position where the text will be placed. 
    The x-coordinate is left_offset, and the y-coordinate is the vertical center of the banner plus an additional 10 pixels. This is done to vertically center the text within the banner."""


    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA) #frame: The image frame where the text will be drawn. cv2.LINE_AA: A flag for anti-aliased line drawing, which makes the text look smoother.


# Initialize background subtractor and morpological operations
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)
""": This function creates a background subtractor object using the K-Nearest Neighbors (KNN) algorithm. The background subtractor is used to separate 
    foreground objects (such as moving objects) from the background in a video sequence.  The history parameter specifies the number of previous frames
    the algorithm will use to build the background model. A higher value means the background model is more stable and less sensitive to short-term changes,
    but it may take longer to adapt to long-term changes in the scene."""
# Kernel size for erosion
ksize = (5, 5) # The kernel is a 5x5 matrix. The choice of kernel size depends on the the size of the noise or objects we want to remove


# Define colors(B,G,R) for drawing bounding boxes and lines.
red = (0, 0, 255) # 'red' will be used to draw bounding boxes around detected motion areas.
blue = (255, 0, 0) # 'blue' will be used to draw horizontal lines in the composite frame.(separation line in  the middle)

# Main processing loop.This loop reads frames from the input video, processes them to detect motion, and writes the processed frames to an output video file.

while True:
    ret, frame = video_cap.read() #video_cap.read() captures a frame from the video. ret is a boolean indicating if the frame was read successfully, and frame is the captured frame.

    if frame is None:  # If the end of the video is reached, and the loop breaks.
        break
    else:
        frame_erode = frame.copy()  # Otherwise, a copy of the frame is made for erosion processing.

    # Stage 1: Motion detection using foreground mask
    fg_mask = bg_sub.apply(frame) #  applies the background subtractor to the frame, producing a foreground mask (fg_mask).
    motion_area = cv2.findNonZero(fg_mask) # finds all non-zero points in the mask, indicating motion.
    x, y, w, h = cv2.boundingRect(motion_area) # computes the bounding box around the detected motion area.

    # Stage 2: Eroded motion detection
    fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8)) #  applies erosion to the foreground mask to remove noise.
    motion_area_erode = cv2.findNonZero(fg_mask_erode) # finds all non-zero points in the eroded mask.
    xe, ye, we, he = cv2.boundingRect(motion_area_erode) # computes the bounding box around the detected motion area in the eroded mask.

    # Draw bounding boxes for detected motion areas
    if motion_area is not None:  # If motion is detected, draw a red bounding box around the motion area in the original frame.
        cv2.rectangle(frame, (x, y), (x + w, y + h), red, thickness=6)
    if motion_area_erode is not None:  # If motion is detected in the eroded mask, draw a red bounding box around the motion area in the eroded frame.
        cv2.rectangle(frame_erode, (xe, ye), (xe + we, ye + he), red, thickness=6)

    # Convert foreground masks to color for visualization
    frame_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) # Convert the grayscale masks to BGR format for better visualization and to build a composite frame with color annotations.
    frame_fg_mask_erode = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)

    # Annotate frames with text
    drawBannerText(frame_fg_mask, 'Foreground Mask')
    drawBannerText(frame_fg_mask_erode, 'Foreground Mask Eroded')

    # Create a composite frame - Stack the frames horizontally (np.hstack) and vertically (np.vstack) to create a composite frame showing the original, eroded, and annotated masks.
    frame_top = np.hstack([frame_fg_mask, frame])
    frame_bot = np.hstack([frame_fg_mask_erode, frame_erode])
    frame_composite = np.vstack([frame_top, frame_bot])

    # Draw a blue line to separate the top and bottom halves of the composite frame
    fc_h, fc_w, _ = frame_composite.shape
    """ The shape attribute of a NumPy array returns a tuple representing the dimensions of the array.
        For a color image, this tuple has three elements: the height (fc_h), the width (fc_w), and the number of color channels."""

    cv2.line(frame_composite, (0, int(fc_h / 2)), (fc_w, int(fc_h / 2)), blue, thickness=1, lineType=cv2.LINE_AA) #draws a horizontal line across the middle of the frame_composite frame using  cv2.line function.
    #(0, int(fc_h / 2)): The starting point of the line
    #0 is the x-coordinate (left edge of the image).
    #int(fc_h / 2) is the y-coordinate (half the height of the image), placing the line in the middle vertically.

    #(fc_w, int(fc_h / 2)): The ending point of the line
    #fc_w is the x-coordinate (right edge of the image).
    #int(fc_h / 2) is the y-coordinate (half the height of the image), same as the starting point y-coordinate, making it a horizontal line.



    # Resize composite frame for compatibility
    frame_composite = cv2.resize(frame_composite, None, fx=0.5, fy=0.5)
    # fx=0.5: The scale factor along the horizontal axis. The width of the frame will be reduced to 50% of its original size.
    #fy=0.5: The scale factor along the vertical axis.  The height of the frame will be reduced to 50% of its original size


    # Write the composite frame to the output video
    video_out_quad.write(frame_composite)

# Release video resources
video_cap.release()
video_out_quad.release()
