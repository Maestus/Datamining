#! /usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = np.array(Image.open("test.png").convert("L"))


import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K=range(1,20)
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
K = 10   http://www.cnblogs.com/wuchuanying/p/6264025.html
"""

