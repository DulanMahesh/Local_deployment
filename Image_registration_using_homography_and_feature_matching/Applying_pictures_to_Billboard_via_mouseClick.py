import cv2
import numpy as np

# Create a named window for the display.
win_name = 'Destination Image'


# ------------------------------------------------------------------------------
# Define mouse callback function.
# ------------------------------------------------------------------------------
def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Render points as yellow circles in destination image.
        cv2.circle(data['img'], (x, y), radius=5, color=(0, 140, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow(win_name, data['img'])
        if len(data['points']) < 4:
            data['points'].append([x, y])


# ------------------------------------------------------------------------------
# Define convenience function for retrieving ROI points in destination image.
# ------------------------------------------------------------------------------
def get_roi_points(img):
    # Set up data to send to mouse handler.
    data = {'img': img.copy(), 'points': []}
    # Set the callback function for any mouse event.
    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, mouse_handler, data)
    cv2.waitKey(0)

    # Convert the list of four separate 2D ROI coordinates to an array.
    roi_points = np.vstack(data['points']).astype(float)

    return roi_points


# ------------------------------------------------------------------------------
# Main processing implementation.
# ------------------------------------------------------------------------------

# Read the source image.
img_src = cv2.imread('Apollo-8-Launch.png')

# Read the destination image.
img_dst = cv2.imread('times_square.jpg')

# Compute the coordinates for the four corners of the source image.
size = img_src.shape
src_pts = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)

# Get four corners of the billboard
print('Click on four corners of a billboard and then press ENTER')

# Retrieve the ROI points from the user mouse clicks.
roi_dst = get_roi_points(img_dst)

# Compute the homography.
h, status = cv2.findHomography(src_pts, roi_dst)

# Warp source image onto the destination image.
warped_img = cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))

# Black out polygonal area in destination image.
cv2.fillConvexPoly(img_dst, roi_dst.astype(int), 0, 16)

# Add the warped image to the destination image.
img_dst = img_dst + warped_img

# Display the updated image with the virtual billboard.
cv2.imshow(win_name, img_dst)

# Wait for a key to be pressed to exit.
cv2.waitKey(0)

# Close the window.
cv2.destroyAllWindows()
