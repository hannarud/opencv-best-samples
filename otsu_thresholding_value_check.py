import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy.signal import savgol_filter, find_peaks


parser = argparse.ArgumentParser(description='Code for checking Otru threshold value computation.')
parser.add_argument('--input', help='Path to input image.', default='../data/lena.png')
args = parser.parse_args()

img = cv2.imread(args.input, 0)
if img is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

print("Threshold value for Otsu - custom computation", thresh)

# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print ("Otsu threshold value computed by OpenCV", ret)
