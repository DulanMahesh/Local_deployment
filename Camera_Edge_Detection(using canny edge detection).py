import cv2
import numpy as np

# Capture video from the default camera (0)
source = cv2.VideoCapture(0)

# Create a window to display the video stream
win_name = 'Filter Demo'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Filter modes
PREVIEW = 0  # Preview Mode
CANNY = 1  # Canny Edge Detector
EMBOSS = 2  # Emboss Filter


# Kernel for emboss filter
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])

# Initialize filter mode and result
image_filter = PREVIEW
result = None

while True:
    # Read a frame from the video source
    has_frame, frame = source.read()
    if not has_frame:
        break


    frame = cv2.flip(frame, 1) # Flip the video frame horizontally(1),vertically(0),both vertically and horizontally(-1)

    # Apply the selected filter
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 30, 150)
    elif image_filter == EMBOSS:
        result = cv2.filter2D(frame, -1, kernel_emboss)

    # Display the result
    cv2.imshow(win_name, result)

    # Read the next key press
    key = cv2.waitKey(1)

    # Exit the loop if 'Q', 'q', or 'ESC' is pressed
    if key == ord('Q') or key == ord('q') or key == 27:
        break
    # Change to Canny filter if 'C' or 'c' is pressed
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    # Change to Preview filter if 'P' or 'p' is pressed
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    # Change to Emboss filter if 'E' or 'e' is pressed
    elif key == ord('E') or key == ord('e'):
        image_filter = EMBOSS

# Release the video source and close the window
source.release()
cv2.destroyWindow(win_name)

