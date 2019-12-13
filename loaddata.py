# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:21:44 2019

@author: AmirHossein
"""

import pandas as pd
import os


class Data:
    def __init__(self, datasetname): 
        self.directory = os.getcwd() + ('\\data\\%s\\'%datasetname)              
    def load(self,benchmark,num):
        if benchmark == 'A1':
            address = self.directory + 'A1Benchmark\\real_%s.csv'%num
        elif benchmark == 'A2':
            address = self.directory + 'A2Benchmark\\synthetic_%s.csv'%num            
        elif benchmark == 'A3':
            address = self.directory + 'A3Benchmark\\A3Benchmark-TS%s.csv'%num            
        elif benchmark == 'A4':
            address = self.directory + 'A4Benchmark\\A4Benchmark-TS%s.csv'%num            
        else:
            return "ERROR"  
        print('Loading following database:')
        print(address.split('\\')[-1])
        return pd.read_csv(address)
        
















