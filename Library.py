import logging as log
import os
import warnings

warnings.filterwarnings("ignore")
log.basicConfig(
    filename="log.txt",
    filemode="w",
    format="-+-\n" + "%(message)s\n",
)

import numpy as NP
import pandas as PD
from pandas import DataFrame as DF

NP.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import sklearn as SKL
import tensorflow as TF
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

TF.get_logger().setLevel("ERROR")
TF.autograph.set_verbosity(0)
import json

import keras
from keras import regularizers
from keras.activations import linear, relu, sigmoid
from keras.callbacks import History
from keras.layers import Dense, Input
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.models import Sequential
from keras.optimizers import Adam
