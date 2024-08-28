import cv2
import numpy as np

# Load images from file system.
img1 = cv2.imread('boy.jpg', cv2.IMREAD_COLOR)  # Read the first image.
img2 = cv2.imread('family.jpg', cv2.IMREAD_COLOR)  # Read the second image.

# Display the loaded images in separate windows.
cv2.imshow('Boy', img1)  # Show the first image in a window named 'Boy'.
cv2.waitKey(0)  # Wait indefinitely until a key is pressed.
cv2.imshow('Family', img2)  # Show the second image in a window named 'Family'.
cv2.waitKey(0)  # Wait indefinitely until a key is pressed.
cv2.destroyAllWindows()  # Close all windows after the key press.

# Load the pre-trained deep learning model for face detection.
modelFile = 'model/res10_300x300_ssd_iter_140000.caffemodel'  # Path to the pre-trained model file.
configFile = 'model/deploy.prototxt'  # Path to the configuration file.

# Create a network object using the loaded model and configuration file.
net = cv2.dnn.readNetFromCaffe(prototxt=configFile, caffeModel=modelFile)

# Define a function to apply Gaussian blur to a region of interest (ROI).
def blur(face, factor=3):
    """
    Apply Gaussian blur to a face region with adjustable blurring factor.
    :param face: Region of interest (ROI) to blur.
    :param factor: Factor determining the amount of blur.
    :return: Blurred ROI.
    """
    h, w = face.shape[:2]  # Get the height and width of the ROI.

    # Restrict the blurring factor to be within a reasonable range.
    if factor < 1: factor = 1
    if factor > 5: factor = 5

    # Compute kernel size for Gaussian blur.
    w_k = int(w / factor)
    h_k = int(h / factor)

    # Ensure kernel dimensions are odd numbers for Gaussian blur.
    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1

    # Apply Gaussian blur with calculated kernel size.
    blurred = cv2.GaussianBlur(face, (w_k, h_k), 0)
    return blurred

# Function to detect faces and apply rectangular blur.
def face_blur_rect(image, net, factor=3, detection_threshold=0.9):
    """
    Detect faces in the image and apply a rectangular blur to each detected face.
    :param image: Input image with faces to blur.
    :param net: Pre-trained DNN model for face detection.
    :param factor: Blurring factor for the Gaussian blur.
    :param detection_threshold: Confidence threshold for face detection.
    :return: Image with faces blurred.
    """
    img = image.copy()  # Make a copy of the original image for processing.

    # Convert image to blob format for DNN model.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])

    # Set the input for the DNN model and perform forward pass to get detections.
    net.setInput(blob)
    detections = net.forward()

    (h, w) = img.shape[:2]  # Get the height and width of the image.

    # Iterate over detected faces.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get confidence score for the detection.
        if confidence > detection_threshold:
            # Extract bounding box coordinates and scale them to the original image size.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract face ROI and apply blur.
            face = img[y1:y2, x1:x2]
            face = blur(face, factor=factor)

            # Replace the original face with the blurred face.
            img[y1:y2, x1:x2] = face

    return img

# Apply rectangular blur to faces in the first image and display results.
img1_rect = face_blur_rect(img1, net, factor=2.5)
cv2.imshow('Original :: Rectangular Blur', cv2.hconcat([img1, img1_rect]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply rectangular blur to faces in the second image and display results.
img2_rect = face_blur_rect(img2, net, factor=2)
cv2.imshow('Original :: Rectangular Blur', cv2.hconcat([img2, img2_rect]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define a function to apply elliptical blur to face regions.
def face_blur_ellipse(image, net, factor=3, detect_threshold=0.90, write_mask=False):
    """
    Detect faces and apply an elliptical blur to each detected face.
    :param image: Input image with faces to blur.
    :param net: Pre-trained DNN model for face detection.
    :param factor: Blurring factor for Gaussian blur.
    :param detect_threshold: Confidence threshold for face detection.
    :param write_mask: Whether to save the elliptical mask to a file.
    :return: Image with faces blurred and elliptical mask applied.
    """
    img = image.copy()  # Make a copy of the original image for processing.
    img_blur = img.copy()  # Copy for applying blur to faces.

    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)  # Initialize mask for elliptical regions.

    # Convert image to blob format for DNN model.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]  # Get the height and width of the image.

    # Iterate over detected faces.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get confidence score for the detection.
        if confidence > detect_threshold:
            # Extract bounding box coordinates and scale them to the original image size.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # Extract face ROI, apply blur, and replace the original face.
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face = blur(face, factor=factor)
            img_blur[int(y1):int(y2), int(x1):int(x2)] = face

            # Define elliptical parameters based on the bounding box.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0

            # Create an elliptical mask.
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle),
                                          (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the elliptical mask to the image.
            np.putmask(img, elliptical_mask, img_blur)

    # Save the elliptical mask if requested.
    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)

    return img

# Apply elliptical blur to faces in the first image and display results.
img1_ellipse = face_blur_ellipse(img1, net, factor=2.5, write_mask=True)
mask = cv2.imread('elliptical_mask.jpg')  # Load the saved elliptical mask.
cv2.imshow('Original :: Elliptical Mask :: Elliptical Blur', cv2.hconcat([img1, mask, img1_ellipse]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply elliptical blur to faces in the second image and display results.
img2_ellipse = face_blur_ellipse(img2, net, factor=2, write_mask=True)
mask = cv2.imread('elliptical_mask.jpg')  # Load the saved elliptical mask.
cv2.imshow('Original :: Elliptical Mask :: Elliptical Blur', cv2.hconcat([img2, mask, img2_ellipse]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define a function to pixelate a region of interest (ROI).
def pixelate(roi, pixels=16):
    """
    Apply pixelation effect to a region of interest.
    :param roi: Region of interest to pixelate.
    :param pixels: Number of pixels per block in pixelation.
    :return: Pixelated ROI.
    """
    roi_h, roi_w = roi.shape[:2]  # Get the height and width of the ROI.

    if roi_h > pixels and roi_w > pixels:
        # Resize ROI to a smaller size, apply pixelation, and resize back to original dimensions.
        roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
        roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_pixelated = roi  # No pixelation if ROI is too small.

    return roi_pixelated

# Function to detect faces and apply pixelated blur.
def face_blur_pixelate(image, net, detection_threshold=0.9, pixels=10):
    """
    Detect faces in the image and apply pixelation to each detected face.
    :param image: Input image with faces to pixelate.
    :param net: Pre-trained DNN model for face detection.
    :param detection_threshold: Confidence threshold for face detection.
    :param pixels: Number of pixels per block in pixelation.
    :return: Image with pixelated faces.
    """
    img = image.copy()  # Make a copy of the original image for processing.

    # Convert image to blob format for DNN model.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]  # Get the height and width of the image.

    # Iterate over detected faces.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get confidence score for the detection.
        if confidence > detection_threshold:
            # Extract bounding box coordinates and scale them to the original image size.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract face ROI, apply pixelation, and replace the original face.
            face = img[y1:y2, x1:x2]
            face = pixelate(face, pixels=pixels)
            img[y1:y2, x1:x2] = face

    return img

# Apply pixelation to faces in the first image and display results.
img1_pixel = face_blur_pixelate(img1, net, pixels=16)
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur', cv2.hconcat([img1, img1_ellipse, img1_pixel]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply pixelation to faces in the second image and display results.
img2_pixel = face_blur_pixelate(img2, net, pixels=16)
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur', cv2.hconcat([img2, img2_ellipse, img2_pixel]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Function to apply both elliptical blur and pixelation to faces.
def face_blur_ellipse_pixelate(image, net, detect_threshold=0.9, factor=3, pixels=10, write_mask=False):
    """
    Detect faces and apply both elliptical blur and pixelation to each detected face.
    :param image: Input image with faces to blur and pixelate.
    :param net: Pre-trained DNN model for face detection.
    :param detect_threshold: Confidence threshold for face detection.
    :param factor: Blurring factor for Gaussian blur.
    :param pixels: Number of pixels per block in pixelation.
    :param write_mask: Whether to save the elliptical mask to a file.
    :return: Image with elliptical blur and pixelation applied.
    """
    img = image.copy()  # Make a copy of the original image for processing.
    img_out = img.copy()  # Copy for applying both blur and pixelation.
    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)  # Initialize mask for elliptical regions.

    # Convert image to blob format for DNN model.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]  # Get the height and width of the image.

    # Iterate over detected faces.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get confidence score for the detection.
        if confidence > detect_threshold:
            # Extract bounding box coordinates and scale them to the original image size.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # Extract face ROI, apply blur, and then pixelate the blurred face.
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face = blur(face, factor=factor)
            face = pixelate(face, pixels=pixels)
            img_out[int(y1):int(y2), int(x1):int(x2)] = face

            # Define elliptical parameters based on the bounding box.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0

            # Create an elliptical mask.
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle),
                                          (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the elliptical mask to the image.
            np.putmask(img, elliptical_mask, img_out)

    # Save the elliptical mask if requested.
    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)

    return img

# Apply combined elliptical blur and pixelation to faces in the first image and display results.
img1_epb = face_blur_ellipse_pixelate(img1, net, factor=3.5, pixels=15)
img2_epb = face_blur_ellipse_pixelate(img2, net, factor=2, pixels=10)
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur :: Elliptical pixelated',
           cv2.hconcat([img1, img1_ellipse, img1_pixel, img1_epb]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply combined elliptical blur and pixelation to faces in the second image and display results.
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur :: Elliptical pixelated',
           cv2.hconcat([img2, img2_ellipse, img2_pixel, img2_epb]))
cv2.waitKey(0)
cv2.destroyAllWindows()
