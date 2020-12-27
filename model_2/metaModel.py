# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:31:44 2019

@author: espenfb
"""


import detInvModel as dim
import savedRes as sr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import metaRes as mr

idx = pd.IndexSlice

class metaModel(object):
    
    
    def __init__(self, time_data, dirs, meta_data):
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
            
        # Meta data
        for k in meta_data.keys():
            setattr(self, k, meta_data[k])
        
        self.meta_data = meta_data
        
        self.model = dim.deterministicModel(time_data, dirs,
                                            mutables = {self.param: True})
        
    def runMetaModel(self):
        
        self.model.buildModel()
        
        if self.kind == 'relative':
            param = getattr(self.model.instance, self.param)
            if self.index == 'None':
                base_param = param.value
            else: 
                base_param = param[self.index]
        
        for i in self.range:
            
            if self.kind == 'relative':
                value = base_param*(1+i)
            else:
                value = i
                
            param = getattr(self.model.instance, self.param)
            if self.index == 'None':
                param.set_value(value)
                print('Solving for ', self.param, ' = ', getattr(self.model.instance, self.param).value)
            else: 
                param[self.index] = value
                print('Solving when ', self.param, ' for ', self.index, ' = ',
                      getattr(self.model.instance, self.param)[self.index].value)
            
            self.model.solve()
    
            self.model.processResults()
    
            self.model.saveRes(self.res_dir + 'Result' + '_' + self.param + '_' + str(i) + '\\')
            
            
            
        self.meta_res = mr.metaRes(self.res_dir, self.meta_data,
                                   data = self.model.data)