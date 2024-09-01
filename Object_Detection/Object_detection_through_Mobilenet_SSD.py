import cv2  # Import the OpenCV library for computer vision tasks
import numpy as np  # Import NumPy for numerical operations

# Define file paths for the model and configuration files
modelFile = 'ssd_mobilenet_frozen_inference_graph.pb'  # Path to the pre-trained model's weights
configFile = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'  # Path to the model configuration file
classFile = 'coco_class_labels.txt'  # Path to the file containing class labels

# Load the pre-trained model using OpenCV's DNN module
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Read class labels from the file and store them in a list
with open(classFile) as fp:
    labels = fp.read().split('\n')  # Split the file content by newlines to get individual labels
print(sorted(labels))  # Print the sorted list of class labels

# Function to perform object detection on an input image
def detect_objects(net, img):
    """Run object detection over the input image."""
    dim = 300  # Define the dimension to resize the input image (300x300 pixels)
    mean = (0, 0, 0)  # Mean values for image normalization (BGR format)

    # Create a blob from the input image for model processing
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform a forward pass to get the detection results
    objects = net.forward()
    return objects

# Load an image to test object detection
food_img = cv2.imread('fruit-vegetable.jpg')  # Read the input image
food_objects = detect_objects(net, food_img)  # Perform object detection on the image

# Each detected object returns a list with the structure: [classId, score, x, y, w, h]
print(f'Detected {len(food_objects[0][0])} objects (no confidence filtering)')  # Print the number of detected objects
first_detected_obj = food_objects[0][0][0]  # Get the first detected object
print('First object:', first_detected_obj)  # Print details of the first detected object

# Function to draw text labels on the image
def draw_text(im, text, x, y):
    """Draws text label at a given x-y position with a black background."""
    fontface = cv2.FONT_HERSHEY_SIMPLEX  # Define the font type
    font_scale = 0.7  # Define the font scale (size)
    thickness = 1  # Define the thickness of the text

    # Get the size of the text box
    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]  # Get the width and height of the text box
    baseline = textSize[1]  # Get the baseline of the text

    # Draw a filled rectangle behind the text to improve visibility
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
    # Draw the text on top of the rectangle
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

# Function to draw bounding boxes and labels on detected objects
def draw_objects(im, objects, threshold=0.25):
    """Displays a box and text for each detected object exceeding the confidence threshold."""
    rows = im.shape[0]  # Get the number of rows (height) of the image
    cols = im.shape[1]  # Get the number of columns (width) of the image

    # Iterate over each detected object
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])  # Get the class ID of the detected object
        score = float(objects[0, 0, i, 2])  # Get the confidence score of the detection

        # Get the coordinates of the bounding box (normalized to image size)
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        # Check if the detection's confidence score is above the threshold
        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)  # Draw the class label text
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Draw the bounding box. here  2 means that the border will be drawn 2 pixels thick.

    return im  # Return the image with drawn objects

# Apply object detection and draw results on the input image
result = draw_objects(food_img.copy(), food_objects, 0.3)  # Use a threshold of 0.3
cv2.imshow('result', result)  # Display the resulting image with detections
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the displayed window

# SECOND EXAMPLE: Comparing confidence thresholds
traffic_img = cv2.imread('Multi.jpg')  # Load another image for testing
traffic_objects = detect_objects(net, traffic_img)  # Detect objects in the new image

# Draw objects with different confidence thresholds
low = draw_objects(traffic_img.copy(), traffic_objects, 0.0)  # No confidence threshold
mid = draw_objects(traffic_img.copy(), traffic_objects, 0.3)  # Medium confidence threshold
high = draw_objects(traffic_img.copy(), traffic_objects, 0.9)  # High confidence threshold

# Display images with different confidence thresholds
cv2.imshow('Low confidence', low), cv2.waitKey(0), cv2.destroyAllWindows()
cv2.imshow('Mid confidence', mid), cv2.waitKey(0), cv2.destroyAllWindows()
cv2.imshow('High confidence', high), cv2.waitKey(0), cv2.destroyAllWindows()
