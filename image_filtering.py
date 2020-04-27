import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Image filtering tutorial.')
parser.add_argument('--input', help='Path to input image.', default='../data/lena.png')
args = parser.parse_args()

img = cv2.imread(args.input)  # Read image in BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB because we'll plot the image using matplotlib
if img is None:
    print('Could not open or find the image: ', args.input)
    exit(0)


# 2D Convolution
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#d-convolution-image-filtering

kernel = np.ones((5,5),np.float32)/25
blur_2dconv = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur_2dconv),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


# Image Blurring (Image Smoothing)
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#image-blurring-image-smoothing

# 1. Averaging
blur_ave = cv2.blur(img,(5,5))

# 2. Gaussian Filtering
blur_gaus = cv2.GaussianBlur(img,(5,5),0)

# 3. Median Filtering (highly effective in removing salt-and-pepper noise)
blur_median = cv2.medianBlur(img,5)

# 4. Bilateral Filtering (highly effective at noise removal while preserving edges)
blur_bilat = cv2.bilateralFilter(img,9,75,75)


# Plot all these filterings
titles = ['Original Image', '2D Convolution', 'Averaging', 'Gaussian Filtering', 'Median Filtering', 'Bilateral Filtering']
images = [img, blur_2dconv, blur_ave, blur_gaus, blur_median, blur_bilat]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
