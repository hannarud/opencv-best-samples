import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy.signal import savgol_filter, find_peaks


parser = argparse.ArgumentParser(description='Code for extracting image background using grayscale histogram. Suitable for images where background is the biggest area of the image.')
parser.add_argument('--input', help='Path to input image.', default='../data/lena.png')
args = parser.parse_args()

img = cv2.imread(args.input, 0)
if img is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_flat = hist.ravel()

hist_flat_smoothed = savgol_filter(hist_flat, 31, 3)
nearly_biggest_height = np.percentile(hist_flat_smoothed, 90)
background_mean = find_peaks(hist_flat_smoothed, height=nearly_biggest_height)[0]

min_nonzero = np.min(np.where(hist_flat > 0))
predicted_sigma = (background_mean - min_nonzero)/3

min_val = background_mean - 3*predicted_sigma
max_val = background_mean + 3*predicted_sigma

plt.bar(np.arange(len(hist_flat)), hist_flat)
plt.plot(hist_flat_smoothed, color='red')
plt.axvline(background_mean, c='red')
plt.axvline(min_val, c='green')
plt.axvline(max_val, c='green')
plt.title("Image grayscale histogram")
plt.show()

print("Mean", background_mean)
print("Sigma", predicted_sigma)

thresh = background_mean + 3*predicted_sigma

# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Global thresholding with thresh value
ret1, th1 = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)

# Plot Otsu and Global with found threshold value
images = [otsu, th1]
titles = ["Otsu's Thresholding",'Global Thresholding (v=found by grayscale histogram)']

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
