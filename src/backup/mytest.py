#! /usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open("test.png").convert("L"))

print(img.shape)

import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

kmeans = KMeans(n_clusters=10)
kmeans.fit(img)
labels = kmeans.labels_
print(len(img[0]))
plt.imshow(img)
plt.show()

"""
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K=range(2,20)
meandistortions=[]
for k in K:
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(img)
	meandistortions.append(sum(np.min(
            cdist(img,kmeans.cluster_centers_,
                 'euclidean'),axis=1))/img.shape[0])

plt.plot(K,meandistortions,'bx-')
plt.show()
"""
"""
cluster1=np.random.uniform(0.5,1.5,(2,10))
print(cluster1)
cluster2=np.random.uniform(3.5,4.5,(2,10))
X=np.hstack((cluster1,cluster2)).T
plt.figure()
plt.axis([0,5,0,5])
plt.grid(True)
plt.plot(X[:,0],X[:,1],'k.')
print(X.shape[0])
print(X.shape[1])
"""