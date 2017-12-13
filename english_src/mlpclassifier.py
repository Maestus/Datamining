# Utilisation de sklearn neural network

import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from loading_dataset import load_dataset

if os.path.isdir('../training_dataset/') :

    images_test, labels_test = load_dataset("../training_dataset/")

    data = images_test.reshape(len(images_test), -1)


    classifier = MLPClassifier(solver="sgd")

    classifier.activation = "logistic"
    classifier.learning_rate_init = 0.001
    classifier.learning_rate = "adaptive"
    #classifier.batch_size = 10

    print(classifier)

    classifier.fit(data, labels_test)

    images_test, labels_test = load_dataset("../test_dataset/", False)

    data_test = images_test.reshape(len(images_test), -1)

    expected = labels_test
    predicted = classifier.predict(data_test)

    accuracy = accuracy_score(expected,predicted)

    print("****************** average_score : " + str(accuracy))
