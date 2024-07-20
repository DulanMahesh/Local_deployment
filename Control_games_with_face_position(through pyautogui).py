#workflow

"""Key Points:
Imports:

cv2 for computer vision tasks.
numpy for numerical operations.
pyautogui for simulating keyboard and mouse actions.
time for time-related functions.
play Function:

1.Captures video, processes frames, and performs actions based on face detection.
detect Function:

2.Uses the pre-trained model to detect faces in each frame.
drawFace Function:

3.Draws rectangles around detected faces for visualization.
checkRect Function:

4.Checks if a detected face is within a predefined bounding box.
move Function:

5.Simulates keyboard movements based on the position of the detected face relative to the bounding box.
Main Loop:

6. Continuously captures frames, detects faces, processes movements, and displays the video feed with annotations."""




import cv2
import numpy as np
import pyautogui as gui
import time

# Set keypress delay to 0 seconds for pyautogui to make keypresses instantaneous.
gui.PAUSE = 0

# Paths to the pre-trained face detection model and its configuration file.
model_path = './model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './model/deploy.prototxt'


def play(prototxt_path, model_path):
    '''
    Run the main loop for capturing video and processing face detection.
    '''
    # Open the video capture stream from the default webcam.
    cap = cv2.VideoCapture(1)

    # Get the width and height of the frames captured by the webcam.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Define the bounding box coordinates centered in the frame.
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    # Ensure the video capture is open before starting the loop.
    while not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while True:
        # Read a frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            return 0  # Exit if no frame is read.

        # Flip the frame horizontally for a mirror effect.
        frame = cv2.flip(frame, 1)

        # To be implemented: Detect faces and draw bounding boxes.
        # Draw the control rectangle on the frame.
        frame = cv2.rectangle(
            frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)

        # Check if the `esc` key is pressed to exit the loop.
        k = cv2.waitKey(5)
        if k == 27:
            return


# Load the pre-trained face detection model from the Caffe framework.
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


def detect(net, frame):
    '''
    Detect faces in the frame.

    returns: list of detected faces, each represented as a dictionary
             with 'start' (x, y) and 'end' (x, y) coordinates and 'confidence'.
    '''
    detected_faces = []
    (h, w) = frame.shape[:2]

    # Prepare the frame for the face detection model.
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # Resize the frame to 300x300.
        1.0,  # Scale factor.
        (300, 300),  # Size of the input image.
        (104.0, 177.0, 123.0))  # Mean subtraction values.

    net.setInput(blob)
    detections = net.forward()  # Perform the forward pass to detect faces.

    # Iterate over detected faces and filter based on confidence level.
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({
                'start': (startX, startY),
                'end': (endX, endY),
                'confidence': confidence})
    return detected_faces


def drawFace(frame, detected_faces):
    '''
    Draw rectangular boxes around detected faces.

    returns: frame with rectangles drawn around detected faces.
    '''
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 10)
    return frame


def checkRect(detected_faces, bbox):
    '''
    Check if any detected face is inside the defined bounding box.

    returns: True if any face is inside the bounding box, otherwise False.
    '''
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        if x1 > bbox[0] and x2 < bbox[1]:
            if y1 > bbox[3] and y2 < bbox[2]:
                return True
    return False


def move(detected_faces, bbox):
    '''
    Simulate keypresses based on the position of detected faces relative to the bounding box.

    The last_mov variable ensures that the character doesn't keep drifting in the same direction
    after the face moves out of the bounding box.
    '''
    global last_mov
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']

        # Check if face is inside the bounding box.
        if checkRect(detected_faces, bbox):
            last_mov = 'center'
            return

        elif last_mov == 'center':
            # Determine direction of movement based on face position.
            if x1 < bbox[0]:
                gui.press('left')
                last_mov = 'left'
            elif x2 > bbox[1]:
                gui.press('right')
                last_mov = 'right'
            if y2 > bbox[2]:
                gui.press('down')
                last_mov = 'down'
            elif y1 < bbox[3]:
                gui.press('up')
                last_mov = 'up'

            # Print the last movement direction.
            if last_mov != 'center':
                print(last_mov)


def play(prototxt_path, model_path):
    '''
    Main loop for capturing video, detecting faces, and controlling based on face position.
    '''
    global last_mov
    prev_frame_time = 0  # Time of the previous frame processing.
    new_frame_time = 0  # Time of the current frame processing.

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(1)  # Open the video capture stream.

    count = 0  # Frame counter.
    init = 0  # Initialization flag.

    # Get the frame width and height.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Define the coordinates of the bounding box centered in the frame.
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    # Ensure the video capture is open before starting the loop.
    while not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while True:
        fps = 0  # Initialize FPS counter.
        ret, frame = cap.read()
        if not ret:
            return 0  # Exit if no frame is read.

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally.

        # Detect faces in the frame.
        detected_faces = detect(net, frame)
        # Draw bounding boxes around detected faces.
        frame = drawFace(frame, detected_faces)
        # Draw the control rectangle on the frame.
        frame = cv2.rectangle(
            frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)

        # Skip every alternate frame to reduce processing load.
        if count % 2 == 0:
            # For the first pass, check if the face is inside the control rectangle.
            if init == 0:
                if checkRect(detected_faces, bbox):
                    init = 1
                    cv2.putText(
                        frame, 'Game is running', (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.waitKey(10)
                    last_mov = 'center'
                    # Click to start the game.
                    gui.click(x=500, y=500)
            else:
                # Simulate keypresses based on face position.
                move(detected_faces, bbox)
                cv2.waitKey(50)

        # Calculate and display FPS.
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        frame = cv2.putText(
            frame, str(fps) + 'FPS', (200, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('camera_feed', frame)
        count += 1

        # Exit the loop on pressing the `esc` key.
        k = cv2.waitKey(5)
        if k == 27:
            return


# Variable to store the last movement direction.
last_mov = ''
play(prototxt_path, model_path)  # Start the main loop.
