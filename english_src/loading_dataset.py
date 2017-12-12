

# Download Full dataset : https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (~1GB)


import os
from shutil import copyfile
from sklearn import datasets, svm, metrics
import fnmatch
import cv2
from skimage import data
import numpy as np
import shutil
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../libs/')
from common import *

path_large_folder = '../by_class/'
path_training_dataset = '../training_dataset/'
nb_letter = 26
nb_number = 10
img_size = 40, 40

def rename_dataset():

    init_f = 3
    init_s = "0"

    new_name = "0"

    index_elem = 0

    nb_elem = (nb_letter * 2) + nb_number

    end = False

    while(nb_elem > index_elem) :

        while((not end) and (index_elem <= nb_elem)) :

            index_elem += 1

            if(index_elem > nb_elem):
                break

            print("Folder " + str(init_f) + str(init_s) + " exist ?" + " " + str(os.path.isdir(path_large_folder + str(init_f) + str(init_s))))

            if(str(os.path.isdir(path_large_folder + str(init_f) + str(init_s)))) :
                print(str(init_f) + str(init_s) + " -> " + new_name)
                os.rename(path_large_folder + str(init_f) + str(init_s), path_large_folder + new_name)

            if(ord(new_name) == 57) :
                new_name = "A"
            elif(ord(new_name) == 90) :
                new_name = "a"
            else :
                new_name = chr(ord(new_name) + 1)


            if(index_elem == nb_number) :
                if(init_f == 3 or init_f == 5) :
                    init_s = "1"
                else :
                    init_s = "0"
                end = True
                break

            if(index_elem == (nb_number + nb_letter)) :
                if(init_f == 3 or init_f == 5) :
                    init_s = "1"
                else :
                    init_s = "0"
                end = True
                break

            if(init_s == "9") :
                init_s = "a"

            elif(init_s == "f") :
                if(init_f == 3 or init_f == 5) :
                    init_s = "1"
                else :
                    init_s = "0"
                end = True
                break

            else :
                init_s = chr(ord(init_s) + 1)


        end = False
        init_f += 1


def get_train_dataset(train_file_nomber) :
    path = os.path.normpath(path_large_folder)

    path = path.rstrip(os.path.sep)
    assert os.path.isdir(path)
    num_sep = path.count(os.path.sep)

    if(os.path.isdir(path_training_dataset)) :
        shutil.rmtree(path_training_dataset)

    os.makedirs(path_training_dataset)

    max_to_read = train_file_nomber

    for root, dirs, files in os.walk(path) :
        if fnmatch.fnmatch(root, '*train*'):
            print(root)
            os.makedirs(path_training_dataset + root.split(os.path.sep)[-2:][0])
            max_to_read = train_file_nomber
            for name in files :
                if(max_to_read > 0):
                    img = Image.open(os.path.join(root, name))
                    img.thumbnail(img_size, Image.ANTIALIAS)
                    img.save(path_training_dataset + root.split(os.path.sep)[-2:][0] + os.path.sep + name)
                    max_to_read -= 1
                else:
                    break
        else:
            continue

def load_dataset():
    #images = []

    h, w = img_size
    images = np.zeros((10*(nb_letter*2+nb_number), h, w, 3), dtype=np.float32)
    labels = []
    im_nb = 0
    for root, dirs, files in os.walk(path_training_dataset) :
        for d in dirs :
            #labels += [d]
            #d_images = []
            for r, dd, imgs  in os.walk(path_training_dataset + str(d)) :
                #feature = np.array([data.imread(path_training_dataset + str(d) + os.path.sep + str(img)) for img in imgs])
                for img in imgs :
                    labels += [d]
                    #d_images += [io.imread(path_training_dataset + str(d) + os.path.sep + str(img))]
                    img_content = data.imread(path_training_dataset + str(d) + os.path.sep + str(img)).astype(np.float64)
                    #img_content = img_content.reshape(len(img_content), -1).astype(np.float64)
                    #img_content = img_content.flatten()
                    #print(img_content.shape)
                    face = np.asarray(img_content, dtype=np.float32)
                    face /= 255.0 # scale uint8 coded colors to the [0.0, 1.0] floats

                    images[im_nb, ...] = face

                    im_nb += 1
                    #images.append(img_content)
                #if(len(d_images) > 0) :
                    #images += [d_images]

    return images, labels


#rename_dataset()
#get_train_dataset(10)
images, labels = load_dataset()
#images = images.reshape(len(images), -1).astype(np.float64)
print(images)

nx = images.shape
print(str(nx))

print(len(labels))

images_and_labels = list(zip(images, labels))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Training: " + str(label))


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)
nx = images.shape
print(str(nx))

data = images.reshape(len(images), -1)
nx = data.shape
print("end " + str(nx))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples], labels[:n_samples])

# Now predict the value of the digit on the second half:
expected = labels[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print(predicted)

images_and_predictions = list(zip(images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Prediction: " + str(prediction))

plt.show()
