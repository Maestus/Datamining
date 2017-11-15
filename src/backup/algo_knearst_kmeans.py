#! /usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = 10