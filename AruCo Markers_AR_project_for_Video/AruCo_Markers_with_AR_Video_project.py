import cv2
import numpy as np
import mimetypes

# Printing and Placement of ArUco Markers to Define a Region of Interest (ROI).
# Initialize the mimetypes module to handle media types.
mimetypes.init()

# Define cases for media manipulation: image in image, image in video, video in image, and video in video.
case1_img_in_img = ['Apollo-8-launch.png', 'office_markers.jpg']
case2_img_in_vid = ['New_Zealand_Cove.jpg', 'office_markers.mp4']
case3_vid_in_img = ['boys_playing.mp4', 'office_markers.jpg']
case4_vid_in_vid = ['horse_race.mp4', 'office_markers.mp4']

# Select the case to be processed (currently set to video in video).
case = case3_vid_in_img

# Define the IDs of the ArUco markers that will be used to define the ROI.
marker_ids = [23, 25, 30, 33]

# Scale factors used to enlarge the size of the source media to cover the ArUco Marker borders.
scaling_fac_x = .008
scaling_fac_y = .012

# Specify the prefix for the output file.
# The output file's media type will depend on the source and destination media type.
prefix = 'AR_'

# Class to specify the source and destination media files.
class MediaSpec:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

# Initialize the MediaSpec object with the selected case's source and destination.
media_spec = MediaSpec(case[0], case[1])

# The source may be either an image or video.
src_input = media_spec.src

# The destination may be either an image or video.
dst_input = media_spec.dst

# Determine the media types for source and destination using mimetypes.
mime_dst = mimetypes.guess_type(dst_input)[0]
if mime_dst is not None:
    mime_dst = mime_dst.split('/')[0]
mime_src = mimetypes.guess_type(src_input)[0]
if mime_src is not None:
    mime_src = mime_src.split('/')[0]

# ------------------------------------------------------------------------------
# Destination (image or video): The media that contains the original scene without any modification.
# ------------------------------------------------------------------------------
if mime_dst == 'image':
    # Read the destination image.
    frame_dst = cv2.imread(dst_input)
elif mime_dst == 'video':
    # Create a video capture object for the destination video.
    cap_dst = cv2.VideoCapture(dst_input)
    fps = cap_dst.get(cv2.CAP_PROP_FPS)  # Frames per second of the video.

# ------------------------------------------------------------------------------
# Source (image or video): The media that will be transformed and placed onto the destination.
# ------------------------------------------------------------------------------
if mime_src == 'image':
    # Read the source image.
    frame_src = cv2.imread(src_input)
elif mime_src == 'video':
    # Create a video capture object for the source video.
    cap_src = cv2.VideoCapture(src_input)
    fps = cap_src.get(cv2.CAP_PROP_FPS)  # Frames per second of the video.

# If either source or destination is a video, create a video writer object.
if mime_dst == 'video' or mime_src == 'video':

    # Determine the output file name based on media types and input file names.
    output_file = prefix + str(mime_src) + '_in_' + str(mime_dst) + '_' + str(src_input[:-4]) + '.mp4'

    if mime_dst == 'video':
        # Determine the output video size based on the destination video frame size.
        width = round(2 * cap_dst.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(cap_dst.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Determine the output video size based on the destination image frame size.
        width = round(2 * frame_dst.shape[1])
        height = round(frame_dst.shape[0])

    # Create the video writer object with the specified file name, codec, FPS, and frame size.
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
else:
    # If both source and destination are images, specify the output file as an image.
    output_file = prefix + 'image_in_image_' + str(src_input[:-4]) + '.jpg'

# Function to extract reference points from marker corners.
def extract_pts(marker_ids, ids, corners):
    # Extract the coordinates of the upper-left corner of the ROI.
    index = np.squeeze(np.where(ids == marker_ids[0]))
    ref_pt1 = np.squeeze(corners[index[0]])[0]

    # Extract the coordinates of the upper-right corner of the ROI.
    index = np.squeeze(np.where(ids == marker_ids[1]))
    ref_pt2 = np.squeeze(corners[index[0]])[1]

    # Extract the coordinates of the lower-right corner of the ROI.
    index = np.squeeze(np.where(ids == marker_ids[2]))
    ref_pt3 = np.squeeze(corners[index[0]])[2]

    # Extract the coordinates of the lower-left corner of the ROI.
    index = np.squeeze(np.where(ids == marker_ids[3]))
    ref_pt4 = np.squeeze(corners[index[0]])[3]

    return ref_pt1, ref_pt2, ref_pt3, ref_pt4

# Function to scale destination points based on specified scaling factors.
def scale_dst_points(ref_pt1, ref_pt2, ref_pt3, ref_pt4, scaling_fac_x=0.01, scaling_fac_y=0.01):
    # Compute horizontal and vertical distances between markers.
    x_distance = np.linalg.norm(ref_pt1 - ref_pt2)  # Distance between upper-left and upper-right markers.
    y_distance = np.linalg.norm(ref_pt1 - ref_pt4)  # Distance between upper-left and lower-left markers.

    # Calculate the pixel adjustments based on scaling factors.
    delta_x = round(scaling_fac_x * x_distance)
    delta_y = round(scaling_fac_y * y_distance)

    # Apply the scaling factors to the ArUco Marker reference points for final destination points.
    pts_dst = [[ref_pt1[0] - delta_x, ref_pt1[1] - delta_y]]
    pts_dst = pts_dst + [[ref_pt2[0] + delta_x, ref_pt2[1] - delta_y]]
    pts_dst = pts_dst + [[ref_pt3[0] + delta_x, ref_pt3[1] + delta_y]]
    pts_dst = pts_dst + [[ref_pt4[0] - delta_x, ref_pt4[1] + delta_y]]

    return pts_dst

# Initialize variables for processing frames.
src_has_frame = True
dst_has_frame = True
frame_count = 0
max_frames = 100
color = (255, 255, 255)  # White color for drawing.

# Load the dictionary that was used to generate the ArUco markers.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Uncomment and modify if custom detector parameters are needed.
# parameters = cv2.aruco.DetectorParameters_create()

# Process Source and Destination Frames.
print('Processing frames, please wait ...')
while src_has_frame & dst_has_frame:

    if mime_dst == 'video':
        # Get frame from the destination video.
        dst_has_frame, frame_dst = cap_dst.read()
        if not dst_has_frame:
            break

    if mime_src == 'video':
        # Get frame from the source video.
        src_has_frame, frame_src = cap_src.read()
        if not src_has_frame:
            break

    # Detect the ArUco markers in the destination image.
    corners, ids, rejected = cv2.aruco.detectMarkers(frame_dst, dictionary)

    # Extract reference point coordinates from the detected marker corners.
    ref_pt1, ref_pt2, ref_pt3, ref_pt4 = extract_pts(marker_ids, ids, corners)

    # Scale the destination points using the provided scaling factors.
    pts_dst = scale_dst_points(ref_pt1, ref_pt2, ref_pt3, ref_pt4,
                               scaling_fac_x=scaling_fac_x,
                               scaling_fac_y=scaling_fac_y)

    # The source points are the four corners of the source image or video frame.
    pts_src = [[0, 0], [frame_src.shape[1], 0], [frame_src.shape[1], frame_src.shape[0]], [0, frame_src.shape[0]]]

    # Convert the list of points to numpy arrays for further processing.
    pts_src_m = np.asarray(pts_src)
    pts_dst_m = np.asarray(pts_dst)

    # Calculate the homography matrix to warp the source frame to the destination's ROI.
    h, mask = cv2.findHomography(pts_src_m, pts_dst_m, cv2.RANSAC)

    # Warp the source image onto the destination image using the homography matrix.
    warped_image = cv2.warpPerspective(frame_src, h, (frame_dst.shape[1], frame_dst.shape[0]))

    # Prepare a mask representing the region to copy from the warped image into the destination frame.
    mask = np.zeros([frame_dst.shape[0], frame_dst.shape[1]], dtype=np.uint8)

    # Fill the ROI in the destination frame with white to create the mask.
    cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

    # Copy the mask into 3 channels.
    warped_image = warped_image.astype(float)
    mask3 = np.zeros_like(warped_image)
    for i in range(0, 3):
        mask3[:, :, i] = mask / 255

    # Create a black region in the destination frame's ROI.
    frame_masked = cv2.multiply(frame_dst.astype(float), 1 - mask3)

    # Create the final result by adding the warped image to the masked destination frame.
    frame_out = cv2.add(warped_image, frame_masked)

    # Show the original frame and the new output frame side by side.
    concatenated_output = cv2.hconcat([frame_dst.astype(float), frame_out])

    # Draw a white vertical line that divides the two image frames.
    frame_w = concatenated_output.shape[1]
    frame_h = concatenated_output.shape[0]
    concatenated_output = cv2.line(concatenated_output,
                                   (int(frame_w / 2), 0),
                                   (int(frame_w / 2), frame_h),
                                   color, thickness=8)

    # Create the output file.
    if mime_dst == 'image' and mime_src == 'image':
        # Create output image.
        cv2.imwrite(output_file, concatenated_output.astype(np.uint8))
        break
    else:
        # Create output video.
        video_writer.write(concatenated_output.astype(np.uint8))

    # Resize the frame for display convenience.
    output_frame = cv2.resize(concatenated_output.astype(np.uint8), None, fx=0.4, fy=0.4)
    cv2.imshow('Output', output_frame)
    # Press 'q' to save and exit.
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Clean up and release resources.
cv2.destroyAllWindows()
if 'video_writer' in locals():
    video_writer.release()
    print('Processing completed, video writer released.')
