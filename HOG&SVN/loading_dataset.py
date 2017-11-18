import os
from shutil import copyfile
import fnmatch

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


get_train_dataset(25)
