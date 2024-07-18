
import cv2
import matplotlib.pyplot as plt

# Read the image
sig_org = cv2.imread('signature.jpg', cv2.IMREAD_COLOR)

# Check if the image is loaded properly
if sig_org is None:
    print("Error: Could not load image. Please check the path and filename.")
else:
    print(f"Original image dimensions: {sig_org.shape}")

    # Display the original image
    plt.imshow(sig_org[:, :, ::-1])
    plt.title('Sample Signature')
    plt.show()

    # Define crop indices (adjust these if necessary)
    crop_y1, crop_y2 = 100, 380
    crop_x1, crop_x2 = 150, 780

    # Check if crop indices are within the image dimensions
    if crop_y2 > sig_org.shape[0] or crop_x2 > sig_org.shape[1]:
        print(f"Error: Crop indices are out of bounds. Image dimensions: {sig_org.shape}")
    else:
        # Crop the signature from the original image
        sig = sig_org[crop_y1:crop_y2, crop_x1:crop_x2, :]
        print(f"Cropped image dimensions: {sig.shape}")

        # Display the cropped image
        plt.imshow(sig[:, :, ::-1])
        plt.title('Cropped Signature')
        plt.show()

        # Convert the cropped image to grayscale
        sig_gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
        plt.imshow(sig_gray, cmap='gray')
        plt.title('Gray Scale Signature')
        plt.show()

        # Create a binary (black and white) mask
        ret, alpha_mask = cv2.threshold(sig_gray, 150, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('alpha_mask.jpg', alpha_mask)
        plt.imshow(alpha_mask, cmap='gray')
        plt.title('Alpha Mask')
        plt.show()

        # Create a blue mask for the signature
        blue_mask = sig.copy()
        blue_mask[:, :] = (255, 0, 0)
        plt.imshow(blue_mask[:, :, ::-1])
        plt.title('Blue Mask')
        plt.show()

        # Blend the original signature with the blue mask
        sig_color = cv2.addWeighted(sig, 1, blue_mask, 0.5, 0)
        plt.imshow(sig_color[:, :, ::-1])
        plt.title('Color Signature with Blue Mask')
        plt.show()

        # Split the color channels from the blended image
        b, g, r = cv2.split(sig_color)

        # Print shapes of the channels for verification
        print("Blue channel shape:", b.shape)
        print("Green channel shape:", g.shape)
        print("Red channel shape:", r.shape)

        # Create a list of the four arrays, including the alpha channel as the 4th member
        new = [b, g, r, alpha_mask]

        # Use the merge() function to create a single, multi-channel array
        png = cv2.merge(new)

        # Save the transparent signature as a PNG file
        cv2.imwrite('extracted_sig.png', png)



