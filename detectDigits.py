from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from six.moves import urllib
import pickle

import cv2
import numpy as np

#dataset = datasets.fetch_mldata("MNIST Original")

###
from scipy.io import loadmat
mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
mnist_raw = loadmat(mnist_path)
dataset = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
}

#print(dataset)

dataset = datasets.fetch_mldata("MNIST Original")

#print(dataset)

###
features = np.array(dataset.data, 'int16')
#print(features)
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm='L1')
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()

clf.fit(hog_features, labels)


joblib.dump(clf, "digits_cls.pkl", compress=3)

# Read the input image
im = cv2.imread("test.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

cv2.imshow('gray', im_gray)


# Threshold the image

img = cv2.GaussianBlur(im_gray, (3, 3), 0)
cv2.imshow('blur', img)
im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('thresh1', im_th)

kernel = np.ones((2,2),np.uint8)
im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel, iterations = 1)
cv2.imshow('thresh2', im_th)

# Find contours in the image
_, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
#rects = [cv2.boundingRect(ctr) for ctr in ctrs]
with open('boxes.txt', 'rb') as fp:
    boxes = pickle.load(fp)

rects = boxes
#print(ctrs)
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
i = 0
for rect in rects:
    i = i + 1
    print(rect)
    x1 = rect[0][0]
    print(x1)
    y1 = rect[0][1]
    print(y1)
    x2= rect[3][0]
    y2 = rect[3][1]


    # Draw the rectangles
    cv2.rectangle(im, rect[0], rect[3], (0, 255, 0), 3)

    if y2 < y1:
        s = y2
        l = y1
    else:
        s = y1
        l = y2

    roi = im_th[s+10:l-10, x1+11:x2-11]



    # Resize the image
    img_size = roi.size
    if(img_size > 0) :
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))


        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm = 'L1')

        #predict
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.imshow(str(int(nbr[0])), roi)
        #annotate
        cv2.putText(im, str(int(nbr[0])), ((x1+x2)//2, (y2+y1)//2),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
