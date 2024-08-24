
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import matplotlib
plt.rcParams['image.cmap'] = 'gray'

# Read images.
imagefiles = glob.glob('./scene/*') # read all the files in the folder
imagefiles.sort()
print(imagefiles)

images = []
for filename in imagefiles:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

num_images = len(images)

# Display images.
plt.figure(figsize = [20,10])
num_cols = 4
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.axis('off')
    plt.title(f" Scene {i + 1}")
    plt.imshow(images[i]);

#using the sticher class

stitcher = cv2.Stitcher_create()
status, panorama = stitcher.stitch(images)
if status == 0:
    plt.figure(figsize = [20,10])
    plt.imshow(panorama)
    plt.title("panaroma striched")
    plt.show()

# crop the panaroma as wanted

plt.figure(figsize = [20,10])
plt.imshow(panorama)
cropped_region = panorama[90:867, 1:2000]
plt.imshow(cropped_region);
plt.title("panaroma striched & cropped_region")
plt.show()