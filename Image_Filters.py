import cv2
import matplotlib.pyplot as plt
import numpy as np

#load images
flower    = cv2.imread('Flowers.jpg')
house     = cv2.imread('House.jpg')
monument  = cv2.imread('Monument.jpg')
santorini = cv2.imread('Santorini.jpg')
new_york  = cv2.imread('New_York.jpg')
coast     = cv2.imread('California_Coast.jpg')

#function for displaying original vs filtered picture
def plot(img1, img2):

    fig = plt.figure(figsize = (20,10))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Filtered Image")

    plt.show()


#Implementation of Black and white filter

def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

img = flower
img_bw = bw_filter(img)
plot(img, img_bw)

img = new_york
img_bw = bw_filter(img)
plot(img, img_bw)
#---------------------------------------------------------
#implementation for sopia /vintage filter

def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype = np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia

img = flower
img_sepia = sepia(img) #calling function sepia(img)
plot(img, img_sepia)

#-----------------------------------------------
#Implementation of vignette effect

def vignette(img, level = 2):

    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width/level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/level)

    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask

    return img_vignette

img = flower
img_vignette = vignette(img)
plot(img, img_vignette)

img = img_sepia
img_vignette = vignette(img)
plot(img, img_vignette)

#--------------------------------------------------
#Edge detection filter(using canny algorithm)

img = coast
img_edges = cv2.Canny(img, 100, 200)
plot(img, img_edges)

#Edge detection filter with blur (for beeter edge detection)

img = coast
img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
img_edges = cv2.Canny(img_blur, 100, 200)
plot(img, img_edges)

#-----------------------------------------------
#embosses edges

def embossed_edges(img):

    kernel = np.array([[0, -3, -3],
                       [3,  0, -3],
                       [3,  3,  0]])

    img_emboss = cv2.filter2D(img, -1, kernel=kernel)
    return img_emboss

img = house
img_emboss = embossed_edges(img)
plot(img,img_emboss)

#--------------------------------------------------
#To improve exposure

def bright(img, level):
    img_bright = cv2.convertScaleAbs(img, beta = level)
    return img_bright

img = monument
img_bright = bright(img, 25)
plot(img, img_bright)

#---------------------------------------------
#pencil sketch filter

img = flower
img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
img_sketch_bw, _ = cv2.pencilSketch(img_blur)
plot(img, img_sketch_bw)

img = santorini
img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
img_sketch_bw, img_sketch_color = cv2.pencilSketch(img_blur)
plot(img, img_sketch_bw)


#-----------------------------------------------
#stylization filter (suitable for photos that are not expected in photorealism. )

img = santorini
img_blur = cv2.GaussianBlur(img, (5,5), 0, 0)
img_style = cv2.stylization(img_blur, sigma_s = 40, sigma_r = 0.1)
plot(img, img_style)