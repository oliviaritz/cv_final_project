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

#dataset = datasets.fetch_mldata("MNIST Original")

#print(dataset)

###
features = np.array(dataset["data"], 'int16')
#print(features)
labels = np.array(dataset["target"], 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=11, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm = 'L1')
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()

clf.fit(hog_features, labels)


joblib.dump(clf, "digits_cls.pkl", compress=3)
