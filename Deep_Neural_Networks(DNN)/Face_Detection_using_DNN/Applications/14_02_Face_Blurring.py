import cv2
import numpy as np

# Preview images.
img1 = cv2.imread('boy.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('family.jpg', cv2.IMREAD_COLOR)

# Display.
cv2.imshow('Boy', img1)
cv2.waitKey(0)
cv2.imshow('Family', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the DNN model.
modelFile = '../model/res10_300x300_ssd_iter_140000.caffemodel'
configFile = '../model/deploy.prototxt'

# Read the model and create a network object.
net = cv2.dnn.readNetFromCaffe(prototxt=configFile, caffeModel=modelFile)

# Define a blurring function.
def blur(face, factor=3):
    
    h, w  = face.shape[:2]

    if factor < 1: factor = 1 # Maximum blurring
    if factor > 5: factor = 5 # Minimal blurring
    
    # Kernel size.
    w_k = int(w/factor)
    h_k = int(h/factor)

    # Insure kernel is an odd number.
    if w_k%2 == 0: w_k += 1 
    if h_k%2 == 0: h_k += 1 

    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred

def face_blur_rect(image, net, factor=3, detection_threshold=0.9):
    
    img = image.copy()
        
    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])
    
    # Pass the blob to the DNN model.
    net.setInput(blob)
    
    # Retrieve detections from the DNN model.
    detections = net.forward()
    
    (h, w) = img.shape[:2]
    
    # Process the detetcions.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract the face ROI.
            face = img[y1:y2, x1:x2]

            face = blur(face, factor=factor)

            # Replace the detected face with the blurred one.
            img[y1:y2, x1:x2] = face
            
    return img

img1_rect = face_blur_rect(img1, net, factor=2.5)

# Display.
cv2.imshow('Original :: Rectangular Blur', cv2.hconcat([img1, img1_rect]))
cv2.waitKey(0)
cv2.destroyAllWindows()

img2_rect = face_blur_rect(img2, net, factor=2)

# Display.
cv2.imshow('Original :: Rectangular Blur', cv2.hconcat([img2, img2_rect]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Elliptical face blur
def face_blur_ellipse(image, net, factor=3, detect_threshold=0.90, write_mask=False):
    
    img = image.copy()
    img_blur = img.copy()
    
    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
    
    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_threshold:

            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # The face is defined by the bounding rectangle from the detection.
            face = img[int(y1):int(y2), int(x1):int(x2), :]
           
            # Blur the rectangular area defined by the bounding box.
            face = blur(face, factor=factor)

            # Copy the `blurred_face` to the blurred image.
            img_blur[int(y1):int(y2), int(x1):int(x2), :] = face
            
            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1)/2, y1 + (y2 - y1)/2)
            e_size   = (x2 - x1, y2 - y1)
            e_angle  = 0.0
            
            # Create an elliptical mask. 
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), 
                                                      (255, 255, 255), -1, cv2.LINE_AA)  
            # Apply the elliptical mask
            np.putmask(img, elliptical_mask, img_blur)
            
    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)

    return img

img1_ellipse = face_blur_ellipse(img1, net, factor=2.5, write_mask=True)
mask = cv2.imread('elliptical_mask.jpg')

# Display.
cv2.imshow('Original :: Elliptical Mask :: Elliptical Blur', cv2.hconcat([img1, mask, img1_ellipse]))
cv2.waitKey(0)
cv2.destroyAllWindows()

img2_ellipse = face_blur_ellipse(img2, net, factor=2, write_mask=True)
mask = cv2.imread('elliptical_mask.jpg')

# Display.
cv2.imshow('Original :: Elliptical Mask :: Elliptical Blur', cv2.hconcat([img2, mask, img2_ellipse]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Pixelated Face Blur.
# Define a pixelate function.

def pixelate(roi, pixels=16):
    
    # Size of region to pixelate.
    roi_h, roi_w = roi.shape[:2]
    
    if roi_h > pixels and roi_w > pixels:
        # Resize input ROI to the (small) pixelated size.
        roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)

        # Now enlarge the pixelated ROI to fill the size of the original ROI.
        roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_pixelated = roi

    return roi_pixelated

# Pixelated face blur.
def face_blur_pixelate(image, net, detection_threshold=0.9, pixels=10):
    img = image.copy()
    
    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            face = img[y1:y2, x1:x2]
            face = pixelate(face, pixels=pixels)
            img[y1:y2, x1:x2] = face
            
    return img

img1_pixel = face_blur_pixelate(img1, net, pixels=16)

# Display.
cv2.imshow('Original :: Elliptical Blur :: Pixellated Blur', cv2.hconcat([img1, img1_ellipse, img1_pixel]))
cv2.waitKey(0)
cv2.destroyAllWindows()

img2_pixel = face_blur_pixelate(img2, net, pixels=16)

# Display.
cv2.imshow('Original :: Elliptical Blur :: Pixellated Blur', cv2.hconcat([img2, img2_ellipse, img2_pixel]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Combined: Elliptical, Blurred and Pixelated.
def face_blur_ellipse_pixelate(image, net, detect_threshold=0.9, factor=3, pixels=10, write_mask=False):
    
    img = image.copy()
    img_out = img.copy()
    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
    
    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_threshold:

            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box

            # The face is defined by the bounding rectangle from the detection.
            face = img[int(y1):int(y2), int(x1):int(x2), :]
            
            # Blur the rectangular area defined by the bounding box.
            face = blur(face, factor=factor)
            
            # Pixelate the blurred face.
            face = pixelate(face, pixels=pixels)

            # Copy the blurred/pixelated face to the output image.
            img_out[int(y1):int(y2), int(x1):int(x2), :] = face
            
            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1)/2, y1 + (y2 - y1)/2)
            e_size   = (x2 - x1, y2 - y1)
            e_angle  = 0.0
            
            # Create an elliptical mask. 
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), 
                                                      (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the elliptical mask.
            np.putmask(img, elliptical_mask, img_out)
            
    if write_mask:
        cv2.imwrite('elliptical_mask.jpg', elliptical_mask)
        
    return img

img1_epb = face_blur_ellipse_pixelate(img1, net, factor=3.5, pixels=15)
img2_epb = face_blur_ellipse_pixelate(img2, net, factor=2, pixels=10)

# Display.
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur :: Elliptical pixelated', cv2.hconcat([img1, img1_ellipse, img1_pixel, img1_epb]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display.
cv2.imshow('Original :: Elliptical Blur :: Pixelated Blur :: Elliptical pixelated', cv2.hconcat([img2, img2_ellipse, img2_pixel, img2_epb]))
cv2.waitKey(0)
cv2.destroyAllWindows()
