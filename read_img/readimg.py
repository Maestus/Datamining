#! /usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = np.array(Image.open("test.png").convert("L"))
print(img.shape) 
print(img[400,430])
print(img[10,10])
plt.imshow(img)
plt.show()


"""
greyimg = Image.open("test.png").convert("L")
greyimg.show()
"""
"""
plt.imshow(img)
plt.show()
"""
"""
print(img.shape)  900,1200,3
print(img.dtype)  uint8
print(img.size)	  3240000
print(type(img))  
"""

