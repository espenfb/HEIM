# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:40:21 2017

@author: espenfb
"""

import pandas as pd
import resTools as rt
from dateutil.relativedelta import relativedelta


def processDetRes(obj, model):
    
    obj.res = {}
    sd = obj.start_date
    ed = obj.end_date - relativedelta(hours = 1)
    timeindex =  pd.DatetimeIndex(start= sd, end = ed, freq = 'H')
    
    # Get time dependent hydrogen results
    hydrogen_direct = rt.timeVarToDict(model, model.hydrogen_direct, model.HYDROGEN_PLANTS)
    hydrogen_to_storage = rt.timeVarToDict(model, model.hydrogen_to_storage, model.HYDROGEN_PLANTS)
    hydrogen_from_storage = rt.timeVarToDict(model, model.hydrogen_from_storage, model.HYDROGEN_PLANTS)
    hydrogen_import = rt.timeVarToDict(model, model.hydrogen_import, model.HYDROGEN_PLANTS)
    storage_level = rt.timeVarToDict(model, model.storage_level, model.HYDROGEN_PLANTS)
    storageCap_dual = rt.timeDualToDict(model, model.storageCap, model.HYDROGEN_PLANTS)
    storage_value = rt.timeDualToDict(model, model.storageBalance, model.HYDROGEN_PLANTS)
    hydrogen_price = rt.timeDualToDict(model, model.hydrogenBalance, model.HYDROGEN_PLANTS)
    obj.res['hydrogen'] = pd.DataFrame()
    for i in model.HYDROGEN_PLANTS:
        data_entry = pd.DataFrame()
        data_entry['hydrogen_direct'] = pd.Series(hydrogen_direct[i])
        data_entry['hydrogen_to_storage'] = pd.Series(hydrogen_to_storage[i])
        data_entry['hydrogen_from_storage'] = pd.Series(hydrogen_from_storage[i])
        data_entry['hydrogen_import'] = pd.Series(hydrogen_import[i])
        data_entry['storage_level'] = pd.Series(storage_level[i])
        data_entry['storageCap_dual'] = pd.Series(storageCap_dual[i])
        data_entry['storage_value'] = pd.Series(storage_value[i])
        data_entry['hydrogen_price'] = pd.Series(hydrogen_price[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.res['hydrogen'] = pd.concat([obj.res['hydrogen'],data_entry], axis = 1)
    if len(obj.res['hydrogen'].index) > 0:
        obj.res['hydrogen'].index = timeindex
    
    # Get time dependent node results
    exp = rt.timeVarToDict(model, model.exp, model.NODES)
    imp = rt.timeVarToDict(model, model.imp, model.NODES)
    voltage_angle = rt.timeVarToDict(model, model.voltage_angle, model.NODES)
    rat = rt.timeVarToDict(model, model.rat, model.NORMAL_NODES)
    nodal_price = rt.timeDualToDict(model, model.energyBalance, model.NORMAL_NODES)
    load = rt.timeParamToDict(model, model.Consumer_load, model.CONSUMERS)
    obj.res['bus'] = pd.DataFrame()
    for i in model.NODES:
        data_entry = pd.DataFrame()
        data_entry['exp'] = pd.Series(exp[i])
        data_entry['imp'] = pd.Series(imp[i])
        data_entry['voltage_angle'] = pd.Series(voltage_angle[i])
        if i in model.NORMAL_NODES:
            data_entry['rat'] = pd.Series(rat[i])
            data_entry['nodal_price'] = pd.Series(nodal_price[i])
            data_entry['load'] = pd.Series(load['C'+str(i)])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.res['bus'] = pd.concat([obj.res['bus'],data_entry], axis = 1)
    obj.res['bus'].index = timeindex
    
     # Get time dependent hydro power results
    prod = rt.timeVarToDict(model, model.prod, model.HYDRO_POWER_PLANTS)
    spill = rt.timeVarToDict(model, model.spill, model.HYDRO_POWER_PLANTS)
    res = rt.timeVarToDict(model, model.res, model.HYDRO_POWER_PLANTS)
    water_value = rt.timeDualToDict(model, model.resBalance, model.HYDRO_POWER_PLANTS)
    obj.res['hydro_power'] = pd.DataFrame()
    for i in model.HYDRO_POWER_PLANTS:
        data_entry = pd.DataFrame()
        data_entry['prod'] = pd.Series(prod[i])
        data_entry['spill'] = pd.Series(spill[i])
        data_entry['res'] = pd.Series(res[i])
        data_entry['water_value'] = pd.Series(water_value[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.res['hydro_power'] = pd.concat([obj.res['hydro_power'],data_entry], axis = 1)
    if len(obj.res['hydro_power'].index) > 0:
        obj.res['hydro_power'].index = timeindex
    
    # Get time dependent generator results
    prod = rt.timeVarToDict(model, model.prod, model.WIND_POWER_PLANTS)
    cur = rt.timeVarToDict(model, model.cur, model.WIND_POWER_PLANTS)
    obj.res['wind_power'] = pd.DataFrame()
    for i in model.WIND_POWER_PLANTS:
        data_entry = pd.DataFrame()
    #            if i not in model.HYDRO_POWER_PLANTS:
        data_entry['prod'] = pd.Series(prod[i])
    #            else:
    #                continue
        if i in model.WIND_POWER_PLANTS:
            data_entry['cur'] = pd.Series(cur[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.res['wind_power'] = pd.concat([obj.res['wind_power'],data_entry], axis = 1)
    if len(obj.res['wind_power'].index) > 0:
        obj.res['wind_power'].index = timeindex
        
    
    # Get time dependent branch results
    branch_flow = model.branch_flow.get_values()
    branch_flow_dict = {(i,j) : [branch_flow[t,i,j] for t in model.TIME] for i,j in model.BRANCHES}
    obj.res['branch_flow'] = pd.DataFrame()
    for i,j in model.BRANCHES:
        obj.res['branch_flow'][(i,j)] = pd.Series(branch_flow_dict[i,j])
    obj.res['branch_flow'].index = timeindex
        
        
    # Get penalty data
    obj.res['penalty'] = pd.DataFrame()
    obj.res['penalty']['fill_dev']= pd.Series(model.fill_dev.get_values())
    obj.res['penalty']['drain_dev']= pd.Series(model.drain_dev.get_values())
    
    return

def saveDetRes(obj, save_dir):        

    obj.res['hydrogen'].to_csv(obj.save_dir + 'hydrogen.csv', sep = ',')
    obj.res['bus'].to_csv(obj.save_dir + 'bus.csv', sep = ',')
    obj.res['hydro_power'].to_csv(obj.save_dir + 'hydro_power.csv', sep = ',')
    obj.res['wind_power'].to_csv(obj.save_dir + 'wind_power.csv', sep = ',')
    obj.res['branch_flow'].to_csv(obj.save_dir + 'flow.csv', sep = ',')
    obj.res['penalty'].to_csv(obj.save_dir + 'penalty.csv', sep = ',')
    
def importDetRes(self, import_dir):
    
    self.imp_res = {}
    
    self.imp_res['hydrogen'] = pd.DataFrame().from_csv(import_dir + "hydrogen.csv", header = [0,1])
    self.imp_res['bus'] = pd.DataFrame().from_csv(import_dir + "bus.csv", header = [0,1])
    self.imp_res['hydro_power'] = pd.DataFrame().from_csv(import_dir + "hydro_power.csv", header = [0,1])
    self.imp_res['wind_power'] = pd.DataFrame().from_csv(import_dir + "wind_power.csv", header = [0,1])
    self.imp_res['flow'] = pd.DataFrame().from_csv(import_dir + "flow.csv")
    self.imp_res['penalty'] = pd.DataFrame().from_csv(import_dir + "penalty.csv")