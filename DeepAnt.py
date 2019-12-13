# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:36:45 2019

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
from sklearn.preprocessing import StandardScaler

class DeepAnt:
    def __init__(self,data,Number,P):
        self.d = data
        self.N = Number
        n = int(len(data)*P)
        self.output = (self.Window(data.iloc[:n,:],Number)).sample(frac=1)
        self.testdata = self.Window(data.iloc[n:,:],Number)
    def Window(self,d,N):
        output = []
        for i in range(N,len(d)):
            l = d.iloc[(i-N):i,1].values.tolist()
            l.extend([d.iloc[i,1]])
            l.extend([d.iloc[i,2]])
            output.append(l)
        return pd.DataFrame(output)
      
    def Train(self):
        # Initialising the CNN
        classifier = Sequential()
        # Step 1 - Convolution        
        classifier.add(Conv1D(filters=32, kernel_size=5, input_shape = (self.N,1), activation = 'relu'))
        # Step 2 - Pooling
        classifier.add(MaxPooling1D(pool_size = 5))
        # Adding a second convolutional layer
        classifier.add(Conv1D(filters=32, kernel_size=5, activation = 'relu'))
        classifier.add(MaxPooling1D(pool_size = 5))
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(units = 60, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        # Compiling the CNN
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
        n = int(len(self.output)*.6)
        x_train = self.output.iloc[:n,:-2]
        y_train = self.output.iloc[:n,-2]
        x_test = self.output.iloc[n:,:-2]
        y_test = self.output.iloc[n:,-2]
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        x_train = np.expand_dims(x_train,axis = 2)
        x_test = np.expand_dims(x_test,axis = 2)
        classifier.fit(x_train, y_train,
                  validation_data=(x_test, y_test))
        self.model = classifier
        self.sc = sc
    def Anomaly_detection(self):
        print(-1)
        x = self.testdata.iloc[:,:-2]
        y = self.testdata.iloc[:,-2].values
        x = self.sc.transform(x)
        x = np.expand_dims(x,axis = 2)
        y_pred = self.model.predict(x)
        return abs(y_pred - y)
    
    


