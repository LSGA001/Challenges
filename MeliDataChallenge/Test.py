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

import dask
import dask.dataframe as dd
# If Sample is true only a portion of train.csv will be used

SAMPLE = True
PATH = Path('/home/leonardo/Desktop/Data/sample') if SAMPLE else Path(
    '/home/leonardo/Desktop/Data')
PATH.mkdir(exist_ok=True)

MODELS_PATH = PATH / 'models'
MODELS_PATH.mkdir(exist_ok=True)

def normalize_title(title):
    return unicodedata.normalize('NFKD', title.lower()).encode(
        'ASCII', 'ignore').decode('utf8')

from sklearn.model_selection import train_test_split

if not (PATH / 'train_prepro.csv').exists() or True:
    df = pd.read_csv(PATH / 'train.csv')

    if SAMPLE:
        _, df = train_test_split(df, test_size=int(0.01*len(df)),
            random_state=42, stratify=df.category)

    df['title'] = df.title.apply(normalize_title)
    df = df[~df.title.isna() & (df.title != 'nan') & (df.title != '')]

    df.to_csv(PATH / 'train_prepro.csv', index=False)
