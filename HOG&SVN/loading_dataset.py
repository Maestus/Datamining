import os
from shutil import copyfile
import fnmatch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def get_train_dataset(train_file_nomber) :
    # TO CHANGE
    # Download Full dataset : https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (more than 1GB)
    path = '/Users/bensassi/Downloads/by_class/'
    path = os.path.normpath(path)

    path = path.rstrip(os.path.sep)
    assert os.path.isdir(path)
    num_sep = path.count(os.path.sep)

    target = '../dataset/'
    os.makedirs(target)

    max_to_read = train_file_nomber

    for root, dirs, files in os.walk(path) :
        if fnmatch.fnmatch(root, '*train*'):
            print(root)
            os.makedirs(target + root.split(os.path.sep)[-1:][0])
            max_to_read = train_file_nomber
            for name in files :
                if(max_to_read > 0):
                    copyfile(os.path.join(root, name), target + root.split(os.path.sep)[-1:][0] + os.path.sep + name)
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
                #print(cv2.imread(os.path.join(root, name)))
                #Lfiles += [np.array(Image.open(os.path.join(root, name)).convert("L"))]
                Lfiles = np.append(Lfiles, cv2.imread(os.path.join(root, name)))
                if(label == '9'):
                    label = chr(ord('A') + 1)
                if(label == 'Z'):
                    label = chr(ord('a') + 1)
                Llabels += [label]
                label = chr(ord(label) + 1)
    return obj({'images' : Lfiles, 'labels' : Llabels})


data = load_dataset()
#print(data.images)
