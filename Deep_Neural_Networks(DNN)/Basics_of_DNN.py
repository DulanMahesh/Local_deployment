# Classification using Densenet121 model with OpenCV's DNN module

import cv2
import numpy as np
import glob

# Step 1: Load the Class Names from a .txt file
# This file contains the names of the 1000 classes that the DenseNet121 model can recognize.
with open('input/classification_classes_ILSVRC2012.txt', 'r') as f:
    image_net_names = f.read().split('\n')

# Store the class names, excluding any empty line that might be at the end of the file.
class_names = image_net_names[:-1]

# Print the total number of classes and the name of the first class to verify the loading process.
print(len(class_names), class_names[0])

# Step 2: Load the Pre-trained DenseNet121 Model
# The model configuration (.prototxt) and pre-trained weights (.caffemodel) are loaded from disk.
config_file = 'models/DenseNet_121.prototxt'  # Configuration file that defines the network structure
model_file = 'models/DenseNet_121.caffemodel'  # Pre-trained weights for the network

# Read the network into memory using OpenCV's dnn module.
model = cv2.dnn.readNet(model=model_file, config=config_file, framework='Caffe')

# Step 3: Load the Image to be Classified
# Load an image from disk that will be passed through the network for classification.
tiger_img = cv2.imread('input/image1.jpg')

# Display the image to the user to ensure the correct image has been loaded.
print('Press any key to continue')
cv2.imshow('Image', tiger_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Convert the Image to a Blob
# The image is preprocessed and converted into a blob, a format suitable for the neural network.
blob = cv2.dnn.blobFromImage(image=tiger_img, scalefactor=0.017, size=(224, 224),
                             mean=(104, 117, 123), swapRB=False, crop=False)

# Step 5: Set the Input Blob for the Neural Network
# The blob is set as the input to the network so it can be processed.
model.setInput(blob)

# Step 6: Forward Pass through the Network
# The blob is passed through the network to get the output predictions.
outputs = model.forward()
final_outputs = outputs[0]  # Since there is only one image, we take the first output.

# Step 7: Reshape and Process the Output
# Reshape the output to a 1D array where each entry corresponds to the probability of a class.
final_outputs = final_outputs.reshape(1000, 1)

# Identify the class with the highest probability.
label_id = np.argmax(final_outputs)

# Convert the raw output scores into probabilities using the softmax function.
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

# Print the first 10 probabilities for inspection and the maximum probability.
print(probs[:10])
print("Max probability:", np.max(probs))

# Get the highest probability and convert it to a percentage.
final_prob = np.max(probs) * 100.0

# Map the identified class index to the corresponding class name.
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}%"  # Format the output as "Class Name, Probability%"

# Display the image with the classification result and probability.
cv2.imshow(str(out_text), tiger_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 8: Define a Function to Classify an Image
# This function allows the classification process to be reused for multiple images.
def classify(img):
    image = img.copy()  # Copy the image to avoid modifying the original
    # Convert the image to a blob, adjusting preprocessing parameters as needed.
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    # Set the blob as input to the network.
    model.setInput(blob)
    # Perform a forward pass through the network.
    outputs = model.forward()

    final_outputs = outputs[0]  # Extract the output
    final_outputs = final_outputs.reshape(1000, 1)  # Reshape the output to 1D
    # Identify the class with the highest probability.
    label_id = np.argmax(final_outputs)
    # Convert the output to probabilities.
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    final_prob = np.max(probs) * 100  # Get the highest probability
    # Map the identified class index to the corresponding class name.
    out_name = class_names[label_id]
    out_text = f"{out_name}, {final_prob:.3f}%"  # Format the output text
    return out_text  # Return the result

# Step 9: Classify Multiple Images in a Directory
images = []  # List to store loaded images
imageclasses = []  # List to store classification results

# Load all images from the specified directory and classify them.
for img_path in glob.glob('input/*.jpg'):
    img = cv2.imread(img_path)  # Read each image from disk
    images.append(img)  # Store the image in the list
    print("Classifying " + img_path)  # Print the image being classified
    imageclasses.append(classify(img))  # Classify the image and store the result

# Step 10: Display Each Image with its Classification
for i, image in enumerate(images):
    # Overlay the classification result on the image.
    cv2.putText(image, str(imageclasses[i]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # Display the image with the classification result.
    cv2.imshow(str(imageclasses[i]), image)
    cv2.waitKey(0)  # Wait for a key press to move to the next image
    cv2.destroyAllWindows()  # Close the image window
