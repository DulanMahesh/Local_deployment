import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set up default display settings for Matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 8)  # Set default figure size to 10x8 inches
matplotlib.rcParams['image.cmap'] = 'gray'       # Set default colormap to grayscale for images

# Display the sample picture
image_filename = 'face.jpg'
img = cv2.imread(image_filename)                # Read the input image using OpenCV
plt.imshow(img[:, :, ::-1])                     # Display the image using Matplotlib (convert BGR to RGB)
plt.axis('off')                                 # Turn off axis
plt.show()                                      # Show the image

# Load the face detection model paths
MODEL_PATH = 'model/res10_300x300_ssd_iter_140000.caffemodel'  # Path to the pre-trained Caffe model
CONFIG_PATH = 'model/deploy.prototxt'                         # Path to the model configuration file

# Load the face detection model
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Function for detecting faces in an image
def detect_faces(image, detection_threshold=0.70):
    # Convert image to blob format (necessary preprocessing step for DNN)
    # The blob is a multi-dimensional array with shape [batch_size, num_channels, height, width]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

    # Set the input to the DNN model
    net.setInput(blob)

    # Perform forward pass to get the detections
    detections = net.forward()

    # List to store the bounding boxes of detected faces
    faces = []

    img_h = image.shape[0]  # Image height
    img_w = image.shape[1]  # Image width

    # Loop through the detections
    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:  # Check if detection confidence is above threshold
            left   = detection[3] * img_w       # Calculate left coordinate of bounding box
            top    = detection[4] * img_h       # Calculate top coordinate of bounding box
            right  = detection[5] * img_w       # Calculate right coordinate of bounding box
            bottom = detection[6] * img_h       # Calculate bottom coordinate of bounding box

            face_w = right - left               # Calculate width of bounding box
            face_h = bottom - top               # Calculate height of bounding box

            face_roi = (left, top, face_w, face_h)  # Define region of interest (ROI) for face
            faces.append(face_roi)              # Add the bounding box to the list

    return np.array(faces).astype(int)          # Convert list to numpy array and return

# Check whether faces are detected in the image
faces = detect_faces(img)

img_display = img.copy()                        # Create a copy of the original image for displaying

# Draw rectangles around detected faces
for face in faces:
    cv2.rectangle(img_display, face, (0, 255, 0), 3)  # Draw green rectangle around each face

plt.imshow(img_display[..., ::-1])              # Display the image with rectangles
plt.axis('off')
plt.show()

# Facial landmark detection
# Create the landmark detector instance using Local Binary Features (LBF) method
landmarkDetector = cv2.face.createFacemarkLBF()

# Load the facial landmark model
model = 'model/lbfmodel.yaml'
landmarkDetector.loadModel(model)

# Detect landmarks for detected faces
retval, landmarksList = landmarkDetector.fit(img, faces)  # Fit the model to detected faces

# First index in the list, "landmarksList[:]" refers to a specific array in the list.
print(landmarksList[0].shape)  # Print the shape of the first set of landmarks
print('')
print('                          x   y ')
print('                         -------')

# Print coordinates of the first and last landmark points
print('First Landmark in list: ', landmarksList[0][0][0][0].astype(int),  landmarksList[0][0][0][1].astype(int))
print(' Last Landmark in list: ', landmarksList[0][0][67][0].astype(int), landmarksList[0][0][67][1].astype(int))
print('')
print(landmarksList)  # Print all landmarks

# Display landmarks on the face (indicated by numbers)
img_display = img.copy()                        # Create a copy of the image for display
landmarks = landmarksList[0][0].astype(int)     # Get the landmarks as integer values
print(len(landmarks))                           # Print number of landmarks (should be 68)

# Draw each landmark as a circle and add text index
for idx in range(len(landmarks)):
    cv2.circle(img_display, landmarks[idx], 2, (0, 255, 255), -1)  # Draw small yellow circle for each landmark
    cv2.putText(img_display, "{}".format(idx), landmarks[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                cv2.LINE_AA)  # Label each landmark with its index

plt.figure(figsize=(20, 20))
plt.imshow(img_display[:, :, ::-1])             # Display the image with landmarks
plt.axis('off')
plt.show()

# Draw landmark points (without the numbers on the face)
img_display = img.copy()                        # Create another copy for drawing
for landmarks in landmarksList:
    cv2.face.drawFacemarks(img_display, landmarks, (0, 255, 0))  # Draw green dots for landmarks

plt.imshow(img_display[..., ::-1])              # Show the image with landmarks
plt.axis('off')
plt.show()

# Integrate the implementation with another image
image_filename = 'family.jpg'                   # Filename of a new image
img = cv2.imread(image_filename)                # Read the new image
img_display_faces = img.copy()                  # Create a copy for face detection
img_display_marks = img.copy()                  # Create a copy for landmark detection

# Detect the faces in the new image
faces = detect_faces(img)

if len(faces) > 0:
    # Render bounding boxes around detected faces
    for face in faces:
        cv2.rectangle(img_display_faces, face, (0, 255, 0), 3)

    # Detect the facial landmarks for detected faces
    retval, landmarksList = landmarkDetector.fit(img, faces)

    # Render landmark points on faces
    for landmarks in landmarksList:
        cv2.face.drawFacemarks(img_display_marks, landmarks, (0, 255, 0))

    # Display the images side by side: with faces and with landmarks
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(img_display_faces[..., ::-1])
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img_display_marks[..., ::-1])
    plt.axis('off')
    plt.show()
else:
    print('No faces detected in image.')         # If no faces are found, print a message
