import os
from shutil import copyfile
import fnmatch
import cv2
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
    # Download Full dataset : https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (~1GB)
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


def load_dataset() :
    path = '../dataset/'
    Lfiles = np.array([])
    label = '0'
    Llabels = []
    for root, dirs, files in os.walk(path) :
        if fnmatch.fnmatch(root, '*train*'):
            for name in files :
                print(os.path.join(root, name))
                #Lfiles += [np.array(Image.open(os.path.join(root, name)).convert("L"))]
                Lfiles = np.append(Lfiles, cv2.imread(os.path.join(root, name)))
                if(label == '9'):
                    label = chr(ord('A') + 1)
                if(label == 'Z'):
                    label = chr(ord('a') + 1)
                Llabels += [label]
                label = chr(ord(label) + 1)
    return obj({'images' : Lfiles, 'labels' : Llabels})

#rename_dataset()
#get_train_dataset(1000)
#data = load_dataset()
#print(data.images)
