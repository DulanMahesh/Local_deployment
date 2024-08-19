import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# The Sketcher class is responsible for handling mouse events and drawing on the image.
class Sketcher:
    """OpenCV Utility class for mouse handling."""

    def __init__(self, windowname, dests, colors_func):
        # Initialize the starting point for drawing as None.
        self.prev_pt = None
        # The name of the OpenCV window where the image will be displayed.
        self.windowname = windowname
        # dests contains the images to be displayed: the original image and the mask.
        self.dests = dests
        # colors_func provides the colors to be used for drawing on the image and mask.
        self.colors_func = colors_func
        # A flag to track if the image has been modified.
        self.dirty = False
        # Display the initial images.
        self.show()
        # Set the mouse callback to the on_mouse method to handle mouse events.
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        # Display the original image.
        cv2.imshow(self.windowname, self.dests[0])
        # Display the mask in a separate window.
        cv2.imshow(self.windowname + ": mask", self.dests[1])

    def on_mouse(self, event, x, y, flags, param):
        """Handles mouse movement and events."""
        # Get the current point (x, y) where the mouse event occurred.
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # If the left mouse button is pressed, set prev_pt to the current point.
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            # If the left mouse button is released, reset prev_pt to None.
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            # If the left mouse button is held down and the mouse is moving, draw a line.
            for dst, color in zip(self.dests, self.colors_func()):
                # Draw a line from the previous point to the current point on both the image and the mask.
                cv2.line(dst, self.prev_pt, pt, color, 5)
            # Mark the image as modified.
            self.dirty = True
            # Update the previous point to the current point.
            self.prev_pt = pt
            # Show the updated images.
            self.show()

# Load the image in color mode.
filename = "Lincoln.jpg"
img = cv2.imread(filename, cv2.IMREAD_COLOR)

# Check if the image was loaded successfully.
if img is None:
    print(f'Failed to load image file: {filename}')

# Create a copy of the original image to be used for inpainting.
img_mask = img.copy()
# Create a black image of the same size as the original image to act as the inpainting mask.
inpaintMask = np.zeros(img.shape[:2], np.uint8)
# Instantiate the Sketcher class, which allows the user to draw on the image and mask.
sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((0, 255, 0), 255))

# Enter an infinite loop to handle user input.
while True:
    # Wait indefinitely for a key press.
    ch = cv2.waitKey()
    if ch == 27:  # ESC key to break the loop and exit.
        break
    if ch == ord('t'):
        # If the 't' key is pressed, apply Telea's Fast Marching Method for inpainting.
        t1t = time.time()  # Record the start time.
        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        t2t = time.time()  # Record the end time.
        ttime = t2t - t1t  # Calculate the time taken for the operation.
        # Display the inpainting result using Telea's method.
        cv2.imshow('Inpaint Output using Telea', res)
    if ch == ord('n'):
        # If the 'n' key is pressed, apply the Navier-Stokes method for inpainting.
        t1n = time.time()  # Record the start time.
        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        t2n = time.time()  # Record the end time.
        ntime = t2n - t1n  # Calculate the time taken for the operation.
        # Display the inpainting result using the Navier-Stokes method.
        cv2.imshow('Inpaint Output using NS Technique', res)
    if ch == ord('r'):
        # If the 'r' key is pressed, reset the image and mask to their original states.
        img_mask[:] = img  # Reset the image to the original.
        inpaintMask[:] = 0  # Clear the mask by setting it to black.
        # Display the reset images.
        sketch.show()

# Close all OpenCV windows when the loop ends (when ESC is pressed).
cv2.destroyAllWindows()

