import cv2
import numpy as np

# Initialize the video capture object to use the default camera

s = 1  # Use the default web camera.
video_cap = cv2.VideoCapture(s)

# Define the window name where the camera feed will be displayed.
win_name = 'Camera Preview'
# Create a named window that can be resized.
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained deep neural network model from Caffe framework.
# The first parameter is the path to the .prototxt file, which contains the model architecture.
# The second parameter is the path to the .caffemodel file, which contains the pre-trained weights.
net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt',
                               './model/res10_300x300_ssd_iter_140000.caffemodel')

# Define the mean values for mean subtraction used during model training.
# These values are specific to the dataset and model used for training.
mean = [104, 117, 123]
# Define the scale factor used for scaling pixel values.
scale = 1.0
# Set the input width and height for the network. The model expects a 300x300 image input size.
in_width = 300
in_height = 300

# Set the detection threshold for face detection.
# Detections with confidence scores below this threshold will be ignored.
detection_threshold = 0.5

# Define the settings for drawing text annotations on the frame.
font_style = cv2.FONT_HERSHEY_SIMPLEX  # Choose a simple font style.
font_scale = 0.5  # Set the font scale (size).
font_thickness = 1  # Set the thickness of the text.

# Main loop to continuously capture frames from the camera.
while True:
    has_frame, frame = video_cap.read()  # Capture a frame from the camera.
    if not has_frame:  # If the frame was not captured successfully, exit the loop.
        break

    # Get the height and width of the captured frame.
    h = frame.shape[0]
    w = frame.shape[1]

    # Optionally flip the frame horizontally. This is often done to create a mirror effect.
    frame = cv2.flip(frame, 1)

    # Convert the captured frame into a blob, which is a binary large object that can be fed to the neural network.
    # The function scales the image, resizes it, subtracts the mean values, and swaps the color channels.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False,
                                 crop=False)

    # Set the input for the network to be the blob.
    net.setInput(blob)
    # Perform a forward pass to compute the network's output (i.e., the detections).
    detections = net.forward()

    # Loop over each detection to process and draw the results.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Extract the confidence score for the detection.
        if confidence > detection_threshold:  # Only consider detections above the threshold.

            # Compute the coordinates of the bounding box for the detected face.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # Draw a rectangle around the detected face.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle with thickness 2.
            # Create a label with the confidence score.
            label = 'Confidence: %.4f' % confidence
            # Calculate the size of the label for proper placement.
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            # Draw a filled rectangle behind the label for better visibility.
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255),
                          cv2.FILLED)
            # Put the label text on the frame.
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

    # Display the annotated frame in the window.
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)  # Wait for 1 millisecond to check if a key is pressed.
    # Exit the loop if 'Q', 'q', or 'Esc' key is pressed.
    if key == ord('Q') or key == ord('q') or key == 27:
        break

# Release the video capture object and close the window.
video_cap.release()
cv2.destroyWindow(win_name)
