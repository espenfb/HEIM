# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:31:44 2019

@author: espenfb
"""


import detInvModel as dim
import savedRes as sr
import pandas as pd
import numpy as np

class metaModel(object):
    
    
    def __init__(self, time_data, dirs):
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        self.model = dim.deterministicModel(time_data, dirs)
        
        
        
    def runMetaModel(self, param, param_range):
        
        self.param = param
        self.param_range = param_range
        
        for i in param_range:
            
            setattr(self.model.data.parameters,param,i)
            
            self.model.buildModel()
            
            self.model.solve()
    
            self.model.processResults()
    
            self.model.saveRes(self.res_dir + 'Result' + '_' + param + '_' + str(i) + '\\')
            
    def loadRes(self):
        
        self.res = []
        
        for i in self.param_range:
            
            self.res.append(sr.savedRes(self.res_dir + 'Result' + '_' +
                                        self.param + '_' + str(i) + '\\',
                                        data = self.model.data))



    def plotH2Sorce(self):
        
        r = pd.DataFrame()
        for n, i in enumerate(self.res):
            param_val = np.round(self.param_range[n],4)
            r[param_val] = i.getH2SourceBus().sum()
            
        r.T.plot(kind = 'bar')