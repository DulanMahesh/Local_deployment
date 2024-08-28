#Higher level perspective of how this code works
"""
1. Initialization and Setup

The script begins by importing necessary libraries, such as OpenCV for computer vision tasks, numpy for numerical operations, and pygame for audio feedback.
It defines a Plotter class that is used to create real-time visualizations of data, specifically for visualizing the Eye Aspect Ratio (EAR).
The script initializes various parameters, such as paths to pre-trained models for face detection (Caffe model) and facial landmark detection (LBF model), and sets up video capture from a webcam or a video file.

2. Face Detection
A deep neural network (DNN) is used to detect faces in the video frames. The face detection model takes an image frame, processes it, and returns coordinates of detected faces.
The largest face in the frame is chosen as the primary face for further processing.

3. Facial Landmark Detection
Once a face is detected, a landmark detection model identifies key points on the face, focusing on the eyes. This model marks the positions of specific facial features.

4. Eye Aspect Ratio (EAR) Calculation
The script calculates the EAR, a metric used to determine whether the eyes are open or closed. It does this by measuring the distances between specific eye landmarks.
The EAR is calculated for both eyes and averaged to get a single value.

5. Blink Detection
The script tracks the state of the eyes (open or closed) based on EAR values. Thresholds are set to determine when the eyes are closed or open.
A blink is detected when the state transitions from closed to open. Each blink increments a counter.

6. Real-Time Visualization and Feedback
The EAR values are plotted in real-time using the Plotter class, providing a visual representation of eye activity.
The blink count is displayed on the video frame, providing real-time feedback.


7. Loop and Exit Conditions
The script runs in a loop, continuously reading frames from the video capture, processing them for blinks, and updating visualizations until the user interrupts the loop (e.g., by pressing 'q').

8. Cleanup
After the loop exits, the script releases the video capture object and closes any OpenCV windows, ensuring resources are properly released.

"""


import cv2
import numpy as np
import time



# Class to create real-time plot.
class Plotter:
    def __init__(self, plot_width, plot_height, sample_buffer=None, scale_value=1):
        self.scale_value = scale_value
        self.width = plot_width
        self.height = plot_height
        self.color = (0, 255, 0)
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 50
        self.margin_r = 50
        self.margin_u = 50
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer

    # Update new values in plot
    def plot(self, val, label="plot", t1=1, t2=1):

        self.t1 = t1
        self.t2 = t2
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0

        self.plots[label].append(int(val * self.scale_value) / self.scale_value)
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            # Show plot using opencv imshow()
            self.show_plot(label)

    def show_plot(self, label):

        self.plot_canvas = np.zeros((self.height, self.width, 3))

        # Specific vertical scaling to achieve a plot upper limit of 0.5
        scale_h = 8 * self.scale_value * (self.height - self.margin_d - self.margin_u) / self.height
        for j, i in enumerate(np.linspace(0, self.sample_buffer - 2, self.width - self.margin_l - self.margin_r)):
            i = int(i)
            color = (0, 255, 0)
            cv2.line(self.plot_canvas, (j + self.margin_l,
                                        int((self.height - self.margin_d - self.margin_u) + self.margin_u -
                                            self.plots[label][i] * scale_h)),
                     (j + self.margin_l,
                      int((self.height - self.margin_d - self.margin_u) + self.margin_u - self.plots[label][
                          i + 1] * scale_h)), color, 1)

        # Draw plot border.
        cv2.rectangle(self.plot_canvas, (self.margin_l, self.margin_u),
                      (self.width - self.margin_r, self.height - self.margin_d), (255, 255, 255), 1)
        # Draw grid lines.
        cv2.line(self.plot_canvas,
                 (self.margin_l, int((self.height - self.margin_d - self.margin_u) / 4) + self.margin_u),
                 (self.width - self.margin_r, int((self.height - self.margin_d - self.margin_u) / 4) + self.margin_u),
                 (1, 1, 1), 1)
        cv2.line(self.plot_canvas,
                 (self.margin_l, int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u),
                 (self.width - self.margin_r, int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u),
                 (1, 1, 1), 1)
        cv2.line(self.plot_canvas,
                 (self.margin_l, int((self.height - self.margin_d - self.margin_u) * 3 / 4) + self.margin_u),
                 (self.width - self.margin_r,
                  int((self.height - self.margin_d - self.margin_u) * 3 / 4) + self.margin_u), (1, 1, 1), 1, )

        # Add y-axis gridline values.
        fontType = cv2.FONT_HERSHEY_TRIPLEX
        font_adjust = 5
        cv2.putText(self.plot_canvas, f"{0.50}", (int(font_adjust), int(0) + self.margin_u + font_adjust), fontType,
                    0.5, (255, 255, 255))
        cv2.putText(self.plot_canvas, f"{0.25}", (
        int(font_adjust), int((self.height - self.margin_d - self.margin_u) * 1 / 2) + self.margin_u + font_adjust),
                    fontType, 0.5, (255, 255, 255))
        cv2.putText(self.plot_canvas, f"{0.0}", (int(font_adjust + 21), int((
                                                                                        self.height - self.margin_d - self.margin_u) * 4 / 4) + self.margin_u + font_adjust),
                    fontType, 0.5, (255, 255, 255))

        color = (0, 255, 255)
        cv2.putText(self.plot_canvas, f" {label} : {self.plots[label][-1]}",
                    (int(self.width / 2 - 50), self.height - 20), fontType, 0.6, color)

        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(1)


# ------------------------------------------------------------------------------
# 1. Initializations.
# ------------------------------------------------------------------------------

# Initialize counter for the number of blinks detected.
BLINK = 0

# Model file paths.
# MODEL_PATH = './model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
MODEL_PATH = 'model/res10_300x300_ssd_iter_140000.caffemodel'
CONFIG_PATH = 'model/deploy.prototxt'
LBF_MODEL = 'model/lbfmodel.yaml'

# Create a face detector network instance.
net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# Create the landmark detector instance.
landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_MODEL)

# Initialize video capture object.

cap = cv2.VideoCapture('input-video.mp4')
state_prev = state_curr = 'open'


# ------------------------------------------------------------------------------
# 2. Function definitions.
# ------------------------------------------------------------------------------

# Convenience function for face detection.
def detect_faces(image, detection_threshold=0.70):
    # Convert to blob.
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

    # Pass the blob to the DNN model.
    net.setInput(blob)

    # Retrieve detections from the DNN model.
    detections = net.forward()

    # Create a list to store face bounding boxes.
    faces = []

    img_h = image.shape[0]
    img_w = image.shape[1]

    # Process the detections.
    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left = detection[3] * img_w
            top = detection[4] * img_h
            right = detection[5] * img_w
            bottom = detection[6] * img_h

            face_w = right - left
            face_h = bottom - top

            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)


def get_primary_face(faces, frame_h, frame_w):
    primary_face_index = None
    face_height_max = 0
    for idx in range(len(faces)):
        face = faces[idx]
        # Confirm bounding box of primary face does not exceed frame size.
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]
        if x1 > frame_w or y1 > frame_h or x2 > frame_w or y2 > frame_h:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue

        # Prioritize the face with the maximum height.
        if face[3] > face_height_max:
            primary_face_index = idx
            face_height_max = face[3]

    if primary_face_index is not None:
        primary_face = faces[primary_face_index]
    else:
        primary_face = None

    return primary_face


def visualize_eyes(landmarks):
    for i in range(36, 48):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 255, 0), -1)


# Using this paper: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def get_eye_aspect_ratio(landmarks):
    # Compute the Euclidean distances between the two sets of
    # vertical eye landmarks.
    vert_dist_1right = calculate_distance(landmarks[37], landmarks[41])
    vert_dist_2right = calculate_distance(landmarks[38], landmarks[40])
    vert_dist_1left = calculate_distance(landmarks[43], landmarks[47])
    vert_dist_2left = calculate_distance(landmarks[44], landmarks[46])

    # Compute the Euclidean distance between the horizontal
    # eye landmark coordinates.
    horz_dist_right = calculate_distance(landmarks[36], landmarks[39])
    horz_dist_left = calculate_distance(landmarks[42], landmarks[45])

    # Compute the eye aspect ratio.
    EAR_left = (vert_dist_1left + vert_dist_2left) / (2.0 * horz_dist_left)
    EAR_right = (vert_dist_1right + vert_dist_2right) / (2.0 * horz_dist_right)

    ear = (EAR_left + EAR_right) / 2
    # Return the eye aspect ratio.
    return ear


def calculate_distance(A, B):
    distance = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    return distance





# ------------------------------------------------------------------------------
# 3. Execution logic.
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    frame_count = 0
    frame_calib = 30  # Number of frames to use for threshold calibration.
    sum_ear = 0

    ret, frame = cap.read()
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    # Create real-time plotter object.
    plot_width = 800
    plot_height = 400
    p = Plotter(plot_width, plot_height, sample_buffer=200, scale_value=100)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret != True:
            break
            print('Unable to read frames')

        # Detect Face.
        faces = detect_faces(frame, detection_threshold=.90)

        if len(faces) > 0:

            # Use the primary (largest) face in the frame.
            primary_face = get_primary_face(faces, frame_h, frame_w)

            if primary_face is not None:
                cv2.rectangle(frame, primary_face, (0, 255, 0), 3)

                # Detect Landmarks
                retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

                if retval:
                    # Get the landmarks for the primary face
                    landmarks = landmarksList[0][0]

                    # Display detections.
                    visualize_eyes(landmarks)

                    # Get eye aspect ratio.
                    ear = get_eye_aspect_ratio(landmarks)

                    # Calibrate thresholds based on initial (frame_calib) frames.
                    if frame_count < frame_calib:
                        frame_count += 1
                        sum_ear = sum_ear + ear
                    elif frame_count == frame_calib:
                        frame_count += 1
                        avg_ear = sum_ear / frame_count
                        # Set high threshold to 90% of average EAR.
                        HIGHER_TH = .90 * avg_ear
                        # Set low threshold to 75% of high threshold.
                        LOWER_TH = .70 * HIGHER_TH
                        print("SET EAR HIGH: ", HIGHER_TH)
                        print("SET EAR LOW: ", LOWER_TH)
                    else:
                        # Real time plot for EAR.
                        p.plot(ear, label='EAR')
                        # We register a blink when the eye status transitions from "closed" to "open"
                        if ear < LOWER_TH:
                            state_curr = 'closed'
                            print("state-closed (EAR): ", ear)
                        elif ear > HIGHER_TH:
                            state_curr = 'open'
                        if state_prev == 'closed' and state_curr == 'open':
                            BLINK += 1
                            print("state-open   (EAR): ", ear)
                            print("BLINK DETECTED\n")
                           # if mixer:
                               # play('click.wav')

                        # Update the previous state.
                        state_prev = state_curr

                        cv2.putText(frame, "Blink Counter: {}".format(BLINK), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('Output', frame)
                        k = cv2.waitKey(1)
                        if k == ord('q'):
                            cv2.destroyAllWindows()
                            break
            else:
                print('No valid face detected.')

    cap.release()

