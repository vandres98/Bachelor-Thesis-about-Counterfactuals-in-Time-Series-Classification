#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, GlobalAveragePooling1D, BatchNormalization, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import function

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from alibi.explainers import CEM
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy import stats
from sklearn.neighbors import DistanceMetric
from tslearn.datasets import UCR_UEA_datasets
from tslearn.neighbors import KNeighborsTimeSeries
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False


# In[2]:


from tensorflow import keras
import time


# In[3]:


def ucr_data_loader(dataset):
    """
    Loads datasets from UCR Archive and splits them into X_train, y_train, X_test, y_test
    """
    
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    
    return X_train, y_train, X_test, y_test


# In[4]:


def label_encoder(training_labels, testing_labels):
    """
    normalize labels such that they contain only values between 0 and n-1 classes, e.g: (1,3,3,5) -> (0,1,1,2)
    """
    
    le = preprocessing.LabelEncoder() 
    le.fit(np.concatenate((training_labels, testing_labels), axis=0)) 
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)
    
    return y_train, y_test



# In[5]:

def ecg200_class_name(cl: int) -> str: 
    """
    Returns class string for class number
    """
    
    if cl == 0: class_name = "Normal Heartbeat"
    elif cl == 1: class_name = "Myocardial Infarcion"
    else: class_name = "Unkown Heartbeat"
    
    return class_name

# In[6]:

def test_prediction_check(prediction: int, label: int) -> bool: 
    """
    Returns whether the prediction was correct or not
    """

    return prediction == label
    
# In[7]:

def ecg5000_real_class_name(cl: int) -> str: 
    """
    Returns class string for class number
    """
    
    if cl == 0: class_name = "Normal Heartbeat"
    elif cl == 1: class_name = "R-on-T Premature Ventricular Contraction"
    elif cl == 2: class_name = "Premature Ventricular Contraction"
    elif cl == 3: class_name = "Supraventricular Premature beat"
    else: class_name = "Unknown Beat"
    
    return class_name

# In[8]:

def ecg5000_pred_class_name(cl: int) -> str: 
    """
    Returns class string for class number
    """
    
    if cl == 0: class_name = "Normal Heartbeat"
    elif cl == 1: class_name = "Abnormal Heartbeat"
    else: class_name = "Unknown Beat"
    
    return class_name
# %%
