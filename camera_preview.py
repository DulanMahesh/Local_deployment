import cv2

# Initialize video capture object
source = cv2.VideoCapture(1)  # Use 0 for the default camera

# Create a named window
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Start capturing frames in a loop
while cv2.waitKey(1) != 27:  # Escape key ASCII code
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)

# Release resources and close the window
source.release()
cv2.destroyWindow(win_name)
