import cv2
import sys

# Set the video source (default is 0, which is the default webcam)
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Initialize video capture object
source = cv2.VideoCapture(s)

# Check if the video source is opened successfully
if not source.isOpened():
    print("Error: Could not open video source.")
    sys.exit()

# Get the width and height of the video frames
frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# Create a window to display the video feed
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the video source
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Display the current frame
    cv2.imshow(win_name, frame)

    # Write the frame to the video file
    out.write(frame)

    # Wait for 1 millisecond for a key press
    key = cv2.waitKey(1)

    # If 'Esc' key (ASCII 27) is pressed, break the loop
    if key == 27:  # Escape
        break

# Release the video capture and video write objects and close the display window
source.release()
out.release()
cv2.destroyWindow(win_name)
