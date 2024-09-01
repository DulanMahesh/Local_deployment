import cv2
import numpy as np


# Download model if not present in the folder.
modelFile = 'ssd_mobilenet_frozen_inference_graph.pb'
configFile = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
classFile = 'coco_class_labels.txt'


# Read Tensorflow network.
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Check Class Labels.
with open(classFile) as fp:
    labels = fp.read().split('\n')
print(sorted(labels))


# Detect Objects.
def detect_objects(net, img):
    """Run object detection over the input image."""
    # Blob dimension (dim x dim)
    dim = 300

    mean = (0, 0, 0)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects


food_img = cv2.imread('fruit-vegetable.jpg')
food_objects = detect_objects(net, food_img)

# Each detected object returns a list with the structure of:
# [[[..., classId, score, x, y, w, h]]]
print(f'Detected {len(food_objects[0][0])} objects (no confidence filtering)')
first_detected_obj = food_objects[0][0][0]
print('First object:', first_detected_obj)


# Display single prediction.
def draw_text(im, text, x, y):
    """Draws text label at a given x-y position with a black background."""
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    # Get text size
    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle.
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


# Display all predictions.
def draw_objects(im, objects, threshold=0.25):
    """Displays a box and text for each detected object exceeding the confidence threshold."""
    rows = im.shape[0]
    cols = im.shape[1]

    # For every detected object.
    for i in range(objects.shape[2]):
        # Find the class and confidence.
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        # Check if the detection is of good quality
        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return im


result = draw_objects(food_img.copy(), food_objects, 0.3)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#SECOND EXAMLE
# Comparing confidence thresholds.
traffic_img = cv2.imread('Multi.jpg')
traffic_objects = detect_objects(net, traffic_img)

# Compare displays with low and high confidence thresholds.
low = draw_objects(traffic_img.copy(), traffic_objects, 0.0)
mid = draw_objects(traffic_img.copy(), traffic_objects, 0.3)
high = draw_objects(traffic_img.copy(), traffic_objects, 0.9)

# Display the different thresholds.
cv2.imshow('Low confidence', low),cv2.waitKey(0),cv2.destroyAllWindows()
cv2.imshow('Mid confidence', mid),cv2.waitKey(0),cv2.destroyAllWindows()
cv2.imshow('High confidence', high),cv2.waitKey(0),cv2.destroyAllWindows()


