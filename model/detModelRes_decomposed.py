# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:40:21 2017

@author: espenfb
"""

import pandas as pd
import resTools as rt
import os
#from dateutil.relativedelta import relativedelta


def processInvRes(obj, model):
    
    obj.inv_res = {}
    
    obj.inv_res['plant'] = pd.DataFrame.from_dict(model.available_plants.get_values(),
                              orient = 'index', columns = [str(model.available_plants)])
    obj.inv_res['plant'] = pd.DataFrame.from_dict(model.new_cap.get_values(),
                              orient = 'index', columns = [str(model.new_cap)])
    obj.inv_res['plant'][str(model.Init_cap)] = pd.DataFrame.from_dict(
            model.Init_cap, orient = 'index', columns = [str(model.Init_cap)])
    obj.inv_res['plant'][str(model.retired_cap)] = pd.DataFrame.from_dict(model.retired_cap.get_values(),
                              orient = 'index', columns = [str(model.retired_cap)])
    
    con1 = obj.inv_res['plant'].new_cap > 0.1
    con2 = obj.inv_res['plant'].Init_cap > 0.1
    obj.inv_res['plant'] = obj.inv_res['plant'][con1|con2]
    obj.inv_res['plant'] = obj.inv_res['plant'].round(decimals = 1)
    
    
    obj.inv_res['line'] = pd.DataFrame.from_dict(model.new_branch_cap.get_values(),
                              orient = 'index', columns = [str(model.new_branch_cap)])
    cap = pd.DataFrame(model.Trans_cap.iteritems(), columns = ['Index','MaxCap'])
    cap.set_index('Index', inplace = True)
    obj.inv_res['line'] = pd.concat([obj.inv_res['line'], cap], axis = 1, join = 'inner').dropna()
    obj.inv_res['line']['From'] = [i[1] for i in obj.inv_res['line'].index]
    obj.inv_res['line']['To'] = [i[2] for i in obj.inv_res['line'].index]
    obj.inv_res['line']['Cap'] = obj.inv_res['line']['new_branch_cap']*obj.inv_res['line']['MaxCap']
    
    
def processOprRes(obj, model):
    
    
    obj.opr_res = {}
    
    timeindex =  obj.time
    
    prod = pd.DataFrame.from_dict(model.prod.get_values(),
                                 orient = 'index', columns = [str(model.prod)])
    prod.index = pd.MultiIndex.from_tuples(prod.index, names = ['time','plant'])
    prod = prod.unstack(level = 1).swaplevel(axis = 1)
    
    obj.opr_res['plant'] = prod
    
    cur = pd.DataFrame.from_dict(model.cur.get_values(),
                                 orient = 'index', columns = [str(model.cur)])
    cur.index = pd.MultiIndex.from_tuples(cur.index, names = ['time','plant'])
    cur = cur.unstack(level = 1).swaplevel(axis = 1)
    
    obj.opr_res['plant'] = pd.concat([obj.opr_res['plant'], cur], axis = 1).sort_values(['plant'], axis = 1)
    
    # Get time dependent hydrogen results
    hydrogen_direct = rt.timeVarToDict(model, model.hydrogen_direct, model.HYDROGEN_PLANTS)
    hydrogen_to_storage = rt.timeVarToDict(model, model.to_storage, model.HYDROGEN_PLANTS)
    hydrogen_from_storage = rt.timeVarToDict(model, model.from_storage, model.HYDROGEN_PLANTS)
    hydrogen_import = rt.timeVarToDict(model, model.hydrogen_import, model.HYDROGEN_PLANTS)
    hydrogen_import_ccs = rt.timeVarToDict(model, model.hydrogen_import_ccs, model.HYDROGEN_PLANTS)
    storage_level = rt.timeVarToDict(model, model.storage_level, model.HYDROGEN_PLANTS)
    storageCap_dual = rt.timeDualToDict(model, model.storageEnergyCap, model.HYDROGEN_PLANTS)
    storage_value = rt.timeDualToDict(model, model.storageBalance, model.HYDROGEN_PLANTS)
    hydrogen_price = rt.timeDualToDict(model, model.hydrogenBalance, model.HYDROGEN_PLANTS)
    obj.opr_res['hydrogen'] = pd.DataFrame()
    for i in model.HYDROGEN_PLANTS:
        data_entry = pd.DataFrame()
        data_entry['hydrogen_direct'] = pd.Series(hydrogen_direct[i])
        data_entry['hydrogen_to_storage'] = pd.Series(hydrogen_to_storage[i])
        data_entry['hydrogen_from_storage'] = pd.Series(hydrogen_from_storage[i])
        data_entry['hydrogen_import'] = pd.Series(hydrogen_import[i])
        data_entry['hydrogen_import_ccs'] = pd.Series(hydrogen_import_ccs[i])
        data_entry['storage_level'] = pd.Series(storage_level[i])
        data_entry['storageCap_dual'] = pd.Series(storageCap_dual[i])
        data_entry['storage_value'] = pd.Series(storage_value[i])
        data_entry['hydrogen_price'] = pd.Series(hydrogen_price[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.opr_res['hydrogen'] = pd.concat([obj.opr_res['hydrogen'],data_entry], axis = 1)
    if len(obj.opr_res['hydrogen'].index) > 0:
        obj.opr_res['hydrogen'].index = timeindex
        
        
     # Get time dependent battery results
    to_storage = rt.timeVarToDict(model, model.to_storage, model.BATTERY_PLANTS)
    from_storage = rt.timeVarToDict(model, model.from_storage, model.BATTERY_PLANTS)
    storage_level = rt.timeVarToDict(model, model.storage_level, model.BATTERY_PLANTS)
    obj.opr_res['battery'] = pd.DataFrame()
    for i in model.BATTERY_PLANTS:
        data_entry = pd.DataFrame()
        data_entry['to_storage'] = pd.Series(to_storage[i])
        data_entry['from_storage'] = pd.Series(from_storage[i])
        data_entry['storage_level'] = pd.Series(storage_level[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.opr_res['battery'] = pd.concat([obj.opr_res['battery'],data_entry], axis = 1)
    if len(obj.opr_res['battery'].index) > 0:
        obj.opr_res['battery'].index = timeindex
        

    # Get time dependent node results
    exp = rt.timeVarToDict(model, model.exp, model.NODES)
    imp = rt.timeVarToDict(model, model.imp, model.NODES)
    voltage_angle = rt.timeVarToDict(model, model.voltage_angle, model.NODES)
    rat = rt.timeVarToDict(model, model.rat, model.NODES)
    nodal_price = rt.timeDualToDict(model, model.energyBalance, model.NODES)
    load = rt.timeParamToDict(model, model.Load, model.LOAD)
    obj.opr_res['bus'] = pd.DataFrame()
    for i in model.NODES:
        data_entry = pd.DataFrame()
        data_entry['exp'] = pd.Series(exp[i])
        data_entry['imp'] = pd.Series(imp[i])
        data_entry['voltage_angle'] = pd.Series(voltage_angle[i])
        if i in model.NODES:
            data_entry['rat'] = pd.Series(rat[i])
            data_entry['nodal_price'] = pd.Series(nodal_price[i])
            for j in model.LOAD_AT_NODE[i]:
                data_entry[j] = pd.Series(load[j])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.opr_res['bus'] = pd.concat([obj.opr_res['bus'],data_entry], axis = 1)
    obj.opr_res['bus'].index = timeindex
    
    
    # Get time dependent generator results
    prod = rt.timeVarToDict(model, model.prod, model.WIND_POWER_PLANTS)
    cur = rt.timeVarToDict(model, model.cur, model.WIND_POWER_PLANTS)
    obj.opr_res['wind_power'] = pd.DataFrame()
    for i in model.WIND_POWER_PLANTS:
        data_entry = pd.DataFrame()
    #            if i not in model.HYDRO_POWER_PLANTS:
        data_entry['prod'] = pd.Series(prod[i])
    #            else:
    #                continue
        if i in model.WIND_POWER_PLANTS:
            data_entry['cur'] = pd.Series(cur[i])
        data_entry.columns = pd.MultiIndex.from_product([[i],data_entry.columns])
        obj.opr_res['wind_power'] = pd.concat([obj.opr_res['wind_power'],data_entry], axis = 1)
    if len(obj.opr_res['wind_power'].index) > 0:
        obj.opr_res['wind_power'].index = timeindex
        
    
    # Get time dependent branch results
    branch_flow = model.branch_flow.get_values()
    branch_flow_dict = {(n,i,j) : [branch_flow[t,n,i,j] for t in model.TIME] for n,i,j in model.BRANCHES}
    obj.opr_res['branch_flow'] = pd.DataFrame()
    for n,i,j in model.BRANCHES:
        obj.opr_res['branch_flow'][(n,i,j)] = pd.Series(branch_flow_dict[n,i,j])
    obj.opr_res['branch_flow'].index = timeindex
        
    
    return

def saveInvRes(obj, save_dir): 

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)       
    
    obj.inv_res['plant'].to_csv(save_dir + 'plant.csv', sep = ',')
    obj.inv_res['line'].to_csv(save_dir + 'line.csv', sep = ',')
    
def saveOprRes(obj, save_dir): 
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)  
    
    obj.opr_res['plant'].to_csv(save_dir + 'plant.csv', sep = ',')
    obj.opr_res['hydrogen'].to_csv(save_dir + 'hydrogen.csv', sep = ',')
    obj.opr_res['battery'].to_csv(save_dir + 'battery.csv', sep = ',')
    obj.opr_res['bus'].to_csv(save_dir + 'bus.csv', sep = ',')
    obj.opr_res['wind_power'].to_csv(save_dir + 'wind_power.csv', sep = ',')
    obj.opr_res['branch_flow'].to_csv(save_dir + 'flow.csv', sep = ',')
    