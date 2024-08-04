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

# Create a window to display the video feed
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

frame_count = 0  # Counter to give each frame a unique filename

while True:
    # Read a frame from the video source
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Display the current frame
    cv2.imshow(win_name, frame)

    # Wait for 1 millisecond for a key press
    key = cv2.waitKey(1)

    # If 'Esc' key (ASCII 27) is pressed, break the loop
    if key == 27:  # Escape
        break
    # If 's' key (ASCII 115) is pressed, save the frame to disk
    elif key == ord('s'):
        # Save the current frame with a unique filename
        filename = f'frame_{frame_count:04d}.png'
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
        frame_count += 1  # Increment the frame counter

# Release the video capture object and close the display window
source.release()
cv2.destroyWindow(win_name)
