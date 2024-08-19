import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

class Sketcher:
    """OpenCV Utility class for mouse handling."""

    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])
        cv2.imshow(self.windowname + ": mask", self.dests[1])

    def on_mouse(self, event, x, y, flags, param):
        """Handles mouse movement and events."""
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

# Read image in color mode.
filename = "Car.jpg"
img = cv2.imread(filename, cv2.IMREAD_COLOR)

# If image is not read properly, return an error.
if img is None:
    print(f'Failed to load image file: {filename}')
    exit(1)

# Create a copy of the original image.
img_mask = img.copy()
# Create a black copy of the original image, acts as a mask.
inpaintMask = np.zeros(img.shape[:2], np.uint8)
# Create a sketch using the OpenCV Utility Class: Sketcher.
sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((0, 255, 0), 255))

# Initialize times to None.
ttime = None
ntime = None

while True:
    ch = cv2.waitKey(0)
    if ch == 27:  # ESC key to break
        break
    if ch == ord('t'):
        # Use the algorithm proposed by Alexendra Telea: Fast Marching Method (2004).
        t1t = time.time()
        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        t2t = time.time()
        ttime = t2t - t1t
        cv2.imshow('Inpaint Output using Telea FMM', res)
    if ch == ord('n'):
        # Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting (2001).
        t1n = time.time()
        res = cv2.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        t2n = time.time()
        ntime = t2n - t1n
        cv2.imshow('Inpaint Output using Navier-Stokes', res)
    if ch == ord('r'):
        img_mask[:] = img
        inpaintMask[:] = 0
        sketch.show()

cv2.destroyAllWindows()

# Debugging: Print times to confirm they're set
print(f"Telea runtime: {ttime}")
print(f"Navier-Stokes runtime: {ntime}")

# Only create the plot if both algorithms were run.
if ttime is not None and ntime is not None:
    times = [ttime, ntime]
    methods = ['INPAINT_TELEA', 'INPAINT_NS']

    # Plot size
    fig = plt.figure(figsize=(10, 10))

    # Creating the stacked bar plot
    plt.bar(methods, times, color='blue', width=0.3)

    plt.xlabel('Algorithms')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison of Inpainting Algorithms')
    plt.show()
else:
    print("No algorithms were applied, so no runtime comparison can be made.")
