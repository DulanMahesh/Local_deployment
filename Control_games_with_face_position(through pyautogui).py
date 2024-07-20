import cv2
import numpy as np
import pyautogui as gui
import time

# Set keypress delay to 0.
gui.PAUSE = 0

# Loading the pre-trained face model.
model_path = './model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './model/deploy.prototxt'


def play(prototxt_path, model_path):
    '''
    Run the main loop until cancelled.
    '''
    cap = cv2.VideoCapture(1)

    # Getting the Frame width and height.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Co-ordinates of the bounding box on frame
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    while not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        # To be added: Detecting and drawing bounding box around faces

        # Drawing the control rectangle in the center of the frame.
        frame = cv2.rectangle(
            frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)
        # To be added: Checking for game-start position, and checking to run keyboard press.
        # Exit the loop on pressing the `esc` key.
        k = cv2.waitKey(5)
        if k == 27:
            return

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect(net, frame):
    '''
    Detect the faces in the frame.

    returns: list of faces in the frame
                here each face is a dictionary of format-
                {'start': (startX,startY), 'end': (endX,endY), 'confidence': confidence}
    '''
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
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
    Draw rectangular box over detected faces.

    returns: frame with rectangular boxes over detected faces.
    '''
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 10)
    return frame

def checkRect(detected_faces, bbox):
    '''
    Check for a detected face inside the bounding box at the center of the frame.

    returns: True or False.
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
    Press correct button depending on the position of detected face and bbox.

    The last_mov check is added for making sure the character doesn't keep
    drifting in the previous detection.
    '''
    global last_mov
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']

        # Center
        if checkRect(detected_faces, bbox):
            last_mov = 'center'
            return

        elif last_mov == 'center':
            # Left
            if x1 < bbox[0]:
                gui.press('left')
                last_mov = 'left'
            # Right
            elif x2 > bbox[1]:
                gui.press('right')
                last_mov = 'right'
            # Down
            if y2 > bbox[2]:
                gui.press('down')
                last_mov = 'down'
            # Up
            elif y1 < bbox[3]:
                gui.press('up')
                last_mov = 'up'

            # Print out the button pressed if any.
            if last_mov != 'center':
                print(last_mov)

def play(prototxt_path, model_path):
    '''
    Run the main loop until cancelled.
    '''
    global last_mov
    # Used to record the time when we processed last frame.
    prev_frame_time = 0
    # Used to record the time at which we processed current frame.
    new_frame_time = 0

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(1)

    # Counter for skipping frame.
    count = 0

    # Used to initialize the game.
    init = 0

    # Getting the Frame width and height.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Co-ordinates of the bounding box on frame
    left_x, top_y = frame_width // 2 - 150, frame_height // 2 - 200
    right_x, bottom_y = frame_width // 2 + 150, frame_height // 2 + 200
    bbox = [left_x, right_x, bottom_y, top_y]

    while not cap.isOpened():
        cap = cv2.VideoCapture(1)

    while True:
        fps = 0
        ret, frame = cap.read()

        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        # Detect the face.
        detected_faces = detect(net, frame)
        # Draw bounding box around detected faces.
        frame = drawFace(frame, detected_faces)
        # Drawing the control rectangle in the center of the frame.
        frame = cv2.rectangle(
            frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)

        # Skipping every alternate frame.
        if count % 2 == 0:
            # For first pass.
            if init == 0:
                # If face is inside the control rectangle.
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

                move(detected_faces, bbox)
                cv2.waitKey(50)
        # Calculating the FPS.
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

# Used to pass the previous move of the user to the play() function.
last_mov = ''
play(prototxt_path, model_path)

