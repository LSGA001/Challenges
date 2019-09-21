import sys
import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import numpy as np
import scipy as sp
from scipy import sparse
import IPython
import sklearn
import matplotlib.pyplot as plt
import mglearn

from fastai.basics import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle as pkl

import unicodedata

# Reads the first 1000 data points and prints the first 5.
df = pd.read_csv("/home/leonardo/Desktop/Data/train.csv", index_col=False,
    nrows=200000)
print("\nFirst five data points:\n{}\n".format(df.head()))

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(1)
