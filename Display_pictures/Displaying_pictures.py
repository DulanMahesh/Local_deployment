
import cv2
import matplotlib.pyplot as plt
import matplotlib

# Set the default figure size and colormap
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Load the image
imagePath = 'shapes.jpg'
image = cv2.imread(imagePath)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: The image at path '{imagePath}' could not be loaded.")
else:
    # Convert the image to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display the original image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
    plt.title('Original Image')
    plt.axis('off')  # Hide the axis

    # Display the grayscale image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.imshow(imageGray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')  # Hide the axis

    # Show both images
    plt.show()
