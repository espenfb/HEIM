# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:40:21 2017

@author: espenfb
"""

import pandas as pd
import systemData as sd
import pickle

idx = pd.IndexSlice

def processRes(obj, model):
    
    #model = obj.instance
    
    obj.obj_val = pd.DataFrame()
    obj.obj_val.loc[0,'obj'] = model.obj.expr()
    
    obj.inv_res = pd.DataFrame()
    
    inv_var = ['new_energy', 'new_power','available_plants',
               'retired_cap']
    for v in inv_var:
        v_inst = getattr(model,v)
        a = pd.DataFrame.from_dict(v_inst.get_values(), orient = 'index',
                                columns = [str(v_inst)])
        obj.inv_res = pd.concat([obj.inv_res,a], axis = 1, sort = True)
        
    inv_param = ['Init_power', 'Init_energy']
    for p in inv_param:
        p_val = getattr(model,p)
        a = pd.DataFrame.from_dict(p_val.extract_values(), orient = 'index',
                                columns = [str(p_val)])
        obj.inv_res = pd.concat([obj.inv_res,a], axis = 1, sort = True)
    
    opr_var = ['prod','cur','state','spill',
                 'exp','imp','rat', 'to_storage', 'from_storage', 'storage']
    
    obj.opr_res = pd.DataFrame()
    
    for v in opr_var:
        v_inst = getattr(model,v)
        a = pd.DataFrame.from_dict(v_inst.get_values(), orient = 'index',
                                columns = [str(v_inst)])
        a.index = pd.MultiIndex.from_tuples(a.index,
                                            names = ['time','plant'])
        if not a.empty:
            a = a.unstack(level = [1]).swaplevel(i = 0, j = 1,axis = 1)
            obj.opr_res = pd.concat([obj.opr_res,a], axis = 1, sort = True)
            
    param = ['Load','Inflow','Inflow_ureg', 'Renewable_profile']
        
    for p in param:
        p_val = getattr(model,p)
        a = pd.DataFrame.from_dict(p_val.extract_values(), orient = 'index',
                                columns = [str(p_val)])
        a.index = pd.MultiIndex.from_tuples(a.index,
                                            names = ['time','plant'])
        a = a.unstack(level = [1]).swaplevel(i = 0, j = 1,axis = 1)
                    
        obj.opr_res = pd.concat([obj.opr_res,a], axis = 1, sort = True)
        
        
    duals = ['storageOutPowerCap','storageInPowerCap','storageEnergyCap',
             'storageBalance','energyBalance','renewableBalance']
    dual_values = pd.DataFrame.from_dict(model.dual.items()).set_index(0)
    for d in duals:
        v = getattr(model,d)
        d_val = dual_values.loc[v.itervalues()]
        d_val.index = list(v.iterkeys())
        d_val.rename({1:d},axis=1, inplace = True)
        d_val.index = pd.MultiIndex.from_tuples(d_val.index,
                                            names = ['time','plant'])
        if not d_val.empty:
            d_val = d_val.unstack(level = [1]).swaplevel(i = 0, j = 1,axis = 1)
            obj.opr_res = pd.concat([obj.opr_res,d_val], axis = 1, sort = True)
                  
            
    obj.inv_line = pd.DataFrame.from_dict(model.new_branch.get_values(),
                              orient = 'index', columns = [str(model.new_branch)])
    cap = pd.DataFrame(model.Branch_cap.iteritems(), columns = ['Index','MaxCap'])
    cap.set_index('Index', inplace = True)
    obj.inv_line = pd.concat([obj.inv_line, cap], axis = 1)
    obj.inv_line['From'] = [i[1] for i in obj.inv_line.index]
    obj.inv_line['To'] = [i[2] for i in obj.inv_line.index]
    obj.inv_line['Cap'] = (obj.inv_line['new_branch']*obj.inv_line['MaxCap']).fillna(obj.inv_line['MaxCap'])
    obj.inv_line.index = [i[0] for i in obj.inv_line.index] 

    obj.opr_res.sort_index(axis = 1,inplace=True)
    obj.opr_res.index = pd.date_range(start = obj.start_date, end = obj.end_date, freq = 'h')[obj.opr_res.index]

def saveRes(obj, save_dir):        
    obj.result = pd.HDFStore(save_dir + 'result.h5')
    
    obj.obj_val.to_hdf(obj.result, key = 'obj_val', format = 'table')
    obj.inv_res.to_hdf(obj.result, key = 'inv_res', format = 'table')
    obj.inv_line.to_hdf(obj.result, key = 'inv_line', format = 'table')
    obj.opr_res.to_hdf(obj.result, key = 'opr_res', format = 'table')
    
    obj.result.close()
    
    f = open(save_dir + 'detData' + '.pkl', 'wb')
    pickle.dump(obj.detDataInstance['detData'], f, pickle.HIGHEST_PROTOCOL)
    
    
def importRes(obj, import_dir):
    
    obj.result = pd.HDFStore(import_dir + 'result.h5', mode = 'r+')
    dr_iterator = obj.result.walk()
                
    for (path, subgroups, subkeys) in dr_iterator:
        for k in subkeys:
            if hasattr(obj, k):
                res = getattr(obj, k)
                res = pd.concat([res, obj.result[path + '/' + k]], axis = 0)
                setattr(obj,k,res)
            else:
                res = obj.result[path + '/' + k]
                setattr(obj,k,res)
                
    f = open(import_dir + 'detData' + '.pkl', 'rb')
    obj.detData =  pickle.load(f)

    
class Results(object):
    
    
    def __init__(self, time_data, dirs):
        
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        self.result = pd.HDFStore(self.res_dir + 'result.h5', mode = 'r+')
        dr_iterator = self.result.walk()
                    
        for (path, subgroups, subkeys) in dr_iterator:
            for k in subkeys:
                if hasattr(self, k):
                    res = getattr(self, k)
                    res = pd.concat([res, self.result[path + '/' + k]], axis = 0)
                    setattr(self,k,res)
                else:
                    res = self.result[path + '/' + k]
                    setattr(self,k,res)
                    
        self.data = sd.systemData(dirs)
        
        
    def getInvCost(self):
        
        cost = pd.Series()
        
        seconds = (self.end_date-self.start_date).total_seconds()
        hours = seconds/3600 + 24
        ratio = hours/8760
        
        wind_data = self.data.wind_power
        wind_data.index = ['W' + str(i) for i in wind_data.bus_indx]
    
        cost.loc['p_max'] = wind_data.inv_cost
            
        storage_data = self.data.storage
        storage_data.index = ['S'+ str(i) for i in storage_data.bus_indx]
            
        cost.loc['storage_power_cap'] = storage_data.power_cost
        cost.loc['storage_energy_cap'] = storage_data.energy_cost
        
        return (self.inv_res*cost).sum().sum()*ratio