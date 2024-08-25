import cv2
import numpy as np


# Load the pre-defined ArUco Marker dictionary that has 250 6x6 marker patterns.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate markers with IDs: 23, 25, 30, and 33.
marker_image1 = cv2.aruco.generateImageMarker(dictionary, 23, 200)
marker_image2 = cv2.aruco.generateImageMarker(dictionary, 25, 200)
marker_image3 = cv2.aruco.generateImageMarker(dictionary, 30, 200)
marker_image4 = cv2.aruco.generateImageMarker(dictionary, 33, 200)

# Display the markers.
cv2.imshow('Marker ID: 23', marker_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Marker ID: 25', marker_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Marker ID: 30', marker_image3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Marker ID: 33', marker_image4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the generated markers.
cv2.imwrite("marker_23.png", marker_image1)
cv2.imwrite("marker_25.png", marker_image2)
cv2.imwrite("marker_30.png", marker_image3)
cv2.imwrite("marker_33.png", marker_image4)

# Detect the markers.
# Read input image with the markers.
frame = cv2.imread('marker_23_printed.png')

# Detect the markers in the destination image.
corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary)

cv2.aruco.drawDetectedMarkers(frame, corners, ids)
cv2.imshow('Detected', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply.
# Read input image with the markers.
frame_dst = cv2.imread('office_markers.jpg')

# Note : Resizing for display convenience only, not required in practice.
frame_dst = cv2.resize(frame_dst, None, fx=0.4, fy=0.4)

# Detect the markers in the destination image.
corners, ids, rejected = cv2.aruco.detectMarkers(frame_dst, dictionary)

frame_detetced = frame_dst.copy()

cv2.aruco.drawDetectedMarkers(frame_detetced, corners, ids)
cv2.imshow('Frame Detected', frame_detetced)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the corner points.
# Extract reference point coordinates from marker corners.

# Upper-left corner of ROI.
index = np.squeeze(np.where(ids == 23))
ref_pt1 = np.squeeze(corners[index[0]])[0]

# Upper-right corner of ROI.
index = np.squeeze(np.where(ids == 25))
ref_pt2 = np.squeeze(corners[index[0]])[1]

# Lower-right corner of ROI.
index = np.squeeze(np.where(ids == 30))
ref_pt3 = np.squeeze(corners[index[0]])[2]

# Lower-left corner of ROI.
index = np.squeeze(np.where(ids == 33))
ref_pt4 = np.squeeze(corners[index[0]])[3]

# Scale the ROI points.
# Compute horizontal and vertical distance between markers.
x_distance = np.linalg.norm(ref_pt1 - ref_pt2)
y_distance = np.linalg.norm(ref_pt1 - ref_pt3)

scaling_fac_x = .008  # Scale factor in x (horizontal).
scaling_fac_y = .012  # Scale factor in y (vertical).

delta_x = round(scaling_fac_x * x_distance)
delta_y = round(scaling_fac_y * y_distance)

# Apply the scaling factors to the ArUco Marker reference points to make.
# the final adjustment for the destination points.
pts_dst = [[ref_pt1[0] - delta_x, ref_pt1[1] - delta_y]]
pts_dst = pts_dst + [[ref_pt2[0] + delta_x, ref_pt2[1] - delta_y]]
pts_dst = pts_dst + [[ref_pt3[0] + delta_x, ref_pt3[1] + delta_y]]
pts_dst = pts_dst + [[ref_pt4[0] - delta_x, ref_pt4[1] + delta_y]]

# Define the source image points.
# Read input image with the markers.
frame_src = cv2.imread("Apollo-8-Launch.png")
cv2.imshow('Image', frame_src)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the image corners of the source image.
pts_src = [[0, 0], [frame_src.shape[1], 0], [frame_src.shape[1], frame_src.shape[0]], [0, frame_src.shape[0]]]

pts_src_m = np.asarray(pts_src)
pts_dst_m = np.asarray(pts_dst)

# Calculate the hmography.
h, mask = cv2.findHomography(pts_src_m, pts_dst_m, cv2.RANSAC)

# Warp source image onto the destination image.
warped_image = cv2.warpPerspective(frame_src, h, (frame_dst.shape[1], frame_dst.shape[0]))

warped_image_copy = warped_image.copy()  # Save for display below.

# Prepare a mask representing the region to copy from the warped image into the destination frame.
mask = np.zeros([frame_dst.shape[0], frame_dst.shape[1]], dtype=np.uint8)

# Fill ROI in destination frame with white to create mask.
cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

# Copy the mask into 3 channels.
warped_image = warped_image.astype(float)
mask3 = np.zeros_like(warped_image)
for i in range(0, 3):
    mask3[:, :, i] = mask / 255

# Create inverse mask.
mask3_inv = 1 - mask3

# Create black region in destination frame ROI.
frame_masked = cv2.multiply(frame_dst.astype(float), mask3_inv)

# Create final result by adding the warped image with the masked destination frame.
frame_out = cv2.add(warped_image, frame_masked)

frame_masked = np.uint8(frame_masked)  # For display below.
frame_out = frame_out.astype(np.uint8)  # For display below.

cv2.imshow('Warped Image', warped_image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Mask3 Inverse', mask3_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Destination Image', frame_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Frame Masked', frame_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Final Result', frame_out)
cv2.waitKey(0)
cv2.destroyAllWindows()