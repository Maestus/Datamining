#! /usr/bin/python3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from random import random
import shutil

"""
for root, dirs, files in os.walk("train_set") :
    for d in dirs:
        for root2, dirs2, files2 in os.walk(os.path.join(root, d)):
            for name in files2:
"""
width = 32
height = 32

def d_hash(image_path):
    img = Image.open(image_path)
    small_img = img.resize((width,height)).convert('L')
    avg = sum(list(small_img.getdata()))/(width*height)
    
    str=''.join(map(lambda i: '0' if i<avg else '1', small_img.getdata()))
    return ''.join(map(lambda x:'%x' % int(str[x:x+4],2), range(0,width*height,4)))

def diff_dhash_img(dh1,dh2):
    difference =  (int(dh1, 16))^(int(dh2, 16))
    return (bin(difference)+"").count("1")

test0 = d_hash("test0.png")
test = d_hash("test.png")
test001 = d_hash("test001.png")
test003 = d_hash("test003.png")
test2 = d_hash("test2.png")
print(diff_dhash_img(test,test001))       
print(diff_dhash_img(test001,test003))    
print(diff_dhash_img(test,test003))       
print(diff_dhash_img(test0,test))
