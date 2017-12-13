import os
from shutil import copyfile
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.externals import joblib
import fnmatch
import cv2
import skimage.data as ski
import numpy as np
import shutil
import matplotlib.pyplot as plt
from loading_dataset import *

if not os.path.isfile('classifier.pkl') :

    get_dataset(160)

    get_dataset(20, False)

    images, labels = load_dataset("../training_dataset/")

    data = images.reshape(len(images), -1)

    classifier = svm.SVC(gamma=0.001)

    print("classifier set\n")

    classifier.fit(data, labels)

    print("fitted\n")

    joblib.dump(classifier, 'classifier.pkl')

    print("file dumped\n")

else :

    classifier = joblib.load('classifier.pkl')

print("process prediction")

images_test, labels_test = load_dataset("../test_dataset/", False)

data_test = images_test.reshape(len(images_test), -1)

expected = labels_test
predicted = classifier.predict(data_test)

accuracy = accuracy_score(expected,predicted)

print("****************** average_score : " + str(accuracy))

images_and_predictions = list(zip(images_test, predicted))
for index, (image, prediction) in enumerate(images_and_predictions[80:120]):
    plt.subplot(2, 20, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(str(prediction))

plt.show()
