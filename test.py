# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:42:50 2019

@author: AmirHossein
"""
import numpy as np
import pandas as pd
from loaddata import Data # for loading databases based on benchmark
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
#####################################
from DeepAnt import DeepAnt
#####################################################################################


if __name__ == '__main__':
    data = Data('Yahoo')
    d = data.load('A2',1)
    deepant = DeepAnt(d,50,.9)
    deepant.Train()
    a=deepant.Anomaly_detection() 








