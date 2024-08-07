import cv2
import sys

# Set the video source (default is 0, which is the default webcam)
s = 1
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

# Define the codec for the video file (for MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize recording state and VideoWriter
recording = False
out = None

# Create a window to display the video feed
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the video source
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # Display the current flipped frame
    cv2.imshow(win_name, flipped_frame)

    # If currently recording, write the flipped frame to the video file
    if recording:
        out.write(flipped_frame)

    # Wait for 1 millisecond for a key press
    key = cv2.waitKey(1)

    # If 'Esc' key (ASCII 27) is pressed, break the loop
    if key == 27:  # Escape
        break
    # If 'r' key (ASCII 114) is pressed, toggle recording state
    elif key == ord('r'):
        recording = not recording
        if recording:
            # Start recording
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
            print("Recording started.")
        else:
            # Stop recording
            out.release()
            out = None
            print("Recording stopped.")

# Release the video capture object and close the display window
source.release()
if out:
    out.release()
cv2.destroyWindow(win_name)
