#! /usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = np.array(Image.open("test.png").convert("L"))

img2 = np.array(Image.open("test001.png").convert("L"))

img3 = np.array(Image.open("test002.png").convert("L"))

def squares(lst):
	return [i**2 for i in lst]
def stddev(lst):
    return (sum(squares([i-sum(lst)/len(lst) for i in lst]))/len(lst))**0.5

def stddev2(lst):
    return (sum(squares(lst))/len(lst))**0.5

def diffimg(img1,img2):
	diff = [];
	for i in (0,len(img1)-1):
		diff.append(img1[i]-img2[i])
	return stddev2(diff)
	


def returnList(collection):
	result = []
	for i in collection:
		result.append(i)
	return result;

import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = 10


model = KMeans(n_clusters=K)
model.fit(img)
y = model.labels_

r1 = pd.Series(y).value_counts()
r11 = returnList(r1);
print(r11)
print(sum(r11))
print(stddev(r11))

model2 = KMeans(n_clusters=K)
model2.fit(img2)
y2 = model2.labels_

r2 = pd.Series(y2).value_counts()
r22 = returnList(r2);
print(r22)
print(sum(r22))
print(stddev(r22))

model3 = KMeans(n_clusters=K)
model3.fit(img3)
y3 = model3.labels_

r3 = pd.Series(y3).value_counts()
r33 = returnList(r3);
print(r33)
print(sum(r33))
print(stddev(r33))

print("==========================")
print(diffimg(r11,r22))
print(diffimg(r11,r33))
print(diffimg(r22,r33))
