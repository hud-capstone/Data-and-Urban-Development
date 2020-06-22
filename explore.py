import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import wrangle as wr
import preprocessing as pr

def get_inertia(k, X):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(X)
    return kmeans.inertia_