# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:07:41 2017

@author: espenfb
"""

import pyomo.environ as pe
import os
import detModelRes as dmr
import production_records as pr
import systemData as sd

class deterministicModel(object):
    ''' Deterministic model for regional power system with hydogen loads,
    wind power and hydro power. '''
    
    def __init__(self, time_data, dirs):
        
        for k in time_data.keys():
            setattr(self, k, time_data[k])
         
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        self.timerange = range(int((self.end_date-self.start_date).total_seconds()/3600))
        
        # Import system data
        self.data = sd.systemData(dirs)
        # Import system time series (e.g. price, load, hydro power res curves)
        self.data.importTimeSeries(self.start_date - relativedelta(hours = 1),
                                   self.end_date + relativedelta(hours = 1),
                                   loadScenGen = False)
        
        self.time = pd.date_range(start = self.start_date,
                                  end = self.end_date,
                                  freq = 'H')
        
        print('Building deterministic operation model...')
        self.detModel = buildDetModel()
    
        # Create concrete instance
        self.detDataInstance = detData(self)
        print('Creating LP problem instance...')
        self.detModelInstance = self.detModel.create_instance(
                                data= self.detDataInstance,
                                name="Deterministic operation model",
                                namespace='detData')
        
        # Enable access to duals
        self.detModelInstance.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        
    def solve(self, printOutput = True):
        
        # Connect to solver
        opt = pe.SolverFactory('gurobi', solver_io='python')
    
        if printOutput:
                print('Solving deterministic operation model...')
            
        # Solve model
        self.pyomo_res = opt.solve(self.detModelInstance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True)
        
    def printModel(self, name = 'detInvModel.txt'):
        
        self.detModelInstance.pprint(name)
        
    def printRes(self):
        
        print('Wind power capacity: ')
        for i in self.detModelInstance.WIND_POWER_PLANTS:
            print(i,': ', self.detModelInstance.P_cap_init[i],' + ', '%.2f' %
                  self.detModelInstance.prod_cap_new[i].value)
        
        print()
        
        print('Electrolyser capacity: ')
        for i in self.detModelInstance.HYDROGEN_PLANTS:
            print(i,': ', self.detModelInstance.Elec_cap_init[i],' + ', '%.2f' %
                  self.detModelInstance.elec_cap_new[i].value)
        
        print()
            
        print('Hydrogen storage capacity: ')
        for i in self.detModelInstance.HYDROGEN_PLANTS:
            print(i,': ', self.detModelInstance.Storage_cap_init[i],' + ', '%.2f' %
                  self.detModelInstance.storage_cap_new[i].value)
            
    def processResults(self, printOutput = True):
        ''' Prosessing results from pyomo form to pandas data-frames
        for storing and plotting. '''
        
        if printOutput:
            print('Prosessing results from deteministic model...')
        
        model = self.detModelInstance
        
        dmr.processDetRes(self, model)
        
    def saveRes(self, save_dir):    
        ''' Saving prosessed results.  '''
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        dmr.saveDetRes(self,save_dir)
        
    def importResults(self, import_dir):
        ''' Importing results from files. '''
        
        dmr.importDetRes(self, import_dir)

def buildDetModel():
        m = pe.AbstractModel('detModel')
        
        ##Sets##
        m.TIME = pe.Set(ordered = True)
#        m.FIRST_TIME = pe.Set()
        m.LAST_TIME = pe.Set(ordered = True)
        
        m.NODES = pe.Set(ordered = True)
        m.NORMAL_NODES = pe.Set()
        m.MARKET_NODES = pe.Set()
        m.BRANCHES = pe.Set(dimen = 2)
        m.BRANCHES_AT_NODE = pe.Set(m.NODES, dimen = 2)
#        m.NODES_AT_BRANCH = pe.Set(m.BRANCHES)
        
        m.POWER_PLANTS = pe.Set()
        m.HYDRO_POWER_PLANTS = pe.Set()
        m.WIND_POWER_PLANTS = pe.Set()   
        m.CONSUMERS = pe.Set()
        m.HYDROGEN_PLANTS = pe.Set()
        
        m.GEN_AT_NODE = pe.Set(m.NORMAL_NODES)
        m.LOAD_AT_NODE = pe.Set(m.NORMAL_NODES)
        m.HYDROGEN_AT_NODE = pe.Set(m.NORMAL_NODES)
        
        ##Parameters##
        m.NTime = pe.Param(within = pe.Integers)
        m.Period_ratio = pe.Param(within = pe.NonNegativeReals)
        
        m.External_price = pe.Param(m.TIME, m.MARKET_NODES, within = pe.Reals)
        m.Rationing_cost = pe.Param(within = pe.NonNegativeReals)
        
        m.Consumer_load = pe.Param(m.TIME, m.CONSUMERS, within = pe.NonNegativeReals)
        
        m.H2_storage_eff = pe.Param(within = pe.NonNegativeReals)
        m.H2_direct_eff = pe.Param(within = pe.NonNegativeReals)
        m.Elec_cap_init = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Elec_cap_max = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Elec_cap_cost = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Storage_cap_init = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Storage_cap_max = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Storage_cap_cost = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_import_cost = pe.Param(within = pe.NonNegativeReals)
        m.Initial_storage = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_demand = pe.Param(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.P_cap_init = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.P_cap_max = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.P_cap_cost = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_profile = pe.Param(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Inflow = pe.Param(m.TIME, m.HYDRO_POWER_PLANTS,within = pe.NonNegativeReals)
        m.Inflow_ureg = pe.Param(m.TIME, m.HYDRO_POWER_PLANTS,within = pe.NonNegativeReals)
        
#        m.Inflow_fac = pe.Param(m.POWER_PLANTS,within = pe.NonNegativeReals)
        m.Res_cap = pe.Param(m.HYDRO_POWER_PLANTS,within = pe.NonNegativeReals)
        m.Initial_res = pe.Param(m.HYDRO_POWER_PLANTS,within = pe.NonNegativeReals)
        m.Fill_cost = pe.Param(within = pe.NonNegativeReals)
        m.Drain_cost = pe.Param(within = pe.NonNegativeReals)
        m.Spill_cost = pe.Param(within = pe.NonNegativeReals)
        
        m.Trans_cap = pe.Param(m.BRANCHES,within = pe.NonNegativeReals)
        m.Reactance = pe.Param(m.BRANCHES,within = pe.Reals) # Non-Negative?
        m.Ref_power = pe.Param(within = pe.NonNegativeReals)
        m.Branch_dir_at_node = pe.Param(m.NODES,m.BRANCHES, within = pe.Integers)
#        m.Exp_profile = pe.Param(m.TIME, m.MARKET_NODES,within = pe.NonNegativeReals)
#        m.Imp_profile = pe.Param(m.TIME, m.MARKET_NODES,within = pe.NonNegativeReals)
                
        # Variables
        m.exp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        m.imp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        
        m.prod = pe.Var(m.TIME, m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.prod_cap_new = pe.Var(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.cur = pe.Var(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        
        m.res = pe.Var(m.TIME, m.HYDRO_POWER_PLANTS, within = pe.NonNegativeReals)
        m.spill = pe.Var(m.TIME, m.HYDRO_POWER_PLANTS, within = pe.NonNegativeReals)
        m.fill_dev = pe.Var(m.HYDRO_POWER_PLANTS, within = pe.NonNegativeReals)
        m.drain_dev = pe.Var(m.HYDRO_POWER_PLANTS, within = pe.NonNegativeReals)
        
        m.elec_cap_new = pe.Var(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_direct = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_to_storage = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_from_storage = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_share = pe.Var(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.storage_cap_new = pe.Var(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.storage_level = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.rat = pe.Var(m.TIME, m.NORMAL_NODES, within = pe.NonNegativeReals)
        m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within = pe.Reals)
        m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
        
        
        ## Constraints##
        
        # ALL POWER PLANTS
        def maxProd_rule(m,t,i):
            return m.prod[t,i]  <= m.P_cap_init[i] + m.prod_cap_new[i]
        m.maxProd = pe.Constraint(m.TIME,m.POWER_PLANTS, rule = maxProd_rule)
        
        def maxProdCap_rule(m,i):
            return m.P_cap_init[i] + m.prod_cap_new[i]  <= m.P_cap_max[i]
        m.maxProdCap = pe.Constraint(m.POWER_PLANTS, rule = maxProdCap_rule)
        
        # WIND POWER
        def generationBalance_rule(m,t,i):
            return m.prod[t,i] + m.cur[t,i] == m.Wind_profile[t,i]*(m.P_cap_init[i] + m.prod_cap_new[i])
        m.prodBalance = pe.Constraint(m.TIME,m.WIND_POWER_PLANTS, rule = generationBalance_rule)        
        
        # Hydro power
        def resBalance_rule(m,t,i):
            if t == 0:
                return m.res[t,i] == m.Initial_res[i]*m.Res_cap[i] + m.Inflow[t,i] - m.prod[t,i] - m.spill[t,i]
            else:
                return m.res[t,i] == m.res[t-1,i] + m.Inflow[t,i] - m.prod[t,i] - m.spill[t,i]
        m.resBalance = pe.Constraint(m.TIME, m.HYDRO_POWER_PLANTS, rule = resBalance_rule )
        
        def resCap_rule(m,t,i):
            return m.Res_cap[i]*0.02 <= m.res[t,i] <= m.Res_cap[i]*0.98
        m.resCap = pe.Constraint(m.TIME, m.HYDRO_POWER_PLANTS, rule = resCap_rule)
        
        def spillCap_rule(m,t,i):
            return m.spill[t,i] <= max(m.P_cap_max[i],m.Inflow[t,i])
        m.spillCap = pe.Constraint(m.TIME, m.HYDRO_POWER_PLANTS, rule = spillCap_rule)
        
        def minProd_rule(m,t,i):
            return m.prod[t,i] + m.spill[t,i]  >= m.Inflow_ureg[t,i]
        m.minProd = pe.Constraint(m.TIME,m.HYDRO_POWER_PLANTS, rule = minProd_rule)    
        
        def endRes_rule(m,lt,i):
            return m.res[lt,i]  + m.fill_dev[i] - m.drain_dev[i] == m.Initial_res[i]*m.Res_cap[i]
        m.endRes = pe.Constraint(m.LAST_TIME, m.HYDRO_POWER_PLANTS, rule = endRes_rule )
        
#         Hydrogen plants       
        def storageBalance_rule(m,t,i):
            if t == 0:
                return m.storage_level[t,i] == m.Initial_storage[i]*(m.Storage_cap_init[i] + m.storage_cap_new[i]) + m.hydrogen_to_storage[t,i] - m.hydrogen_from_storage[t,i]
            else:
                return m.storage_level[t,i] == m.storage_level[t-1,i] + m.hydrogen_to_storage[t,i] - m.hydrogen_from_storage[t,i]
        m.storageBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = storageBalance_rule)
        
        def endStorage_rule(m,t,i):
            return m.storage_level[t,i] == m.Initial_storage[i]*(m.Storage_cap_init[i] + m.storage_cap_new[i])
        m.endStorage = pe.Constraint(m.LAST_TIME, m.HYDROGEN_PLANTS, rule = endStorage_rule)
        
        def storageCap_rule(m, t, i):
            return m.storage_level[t,i] <=  m.Storage_cap_init[i] + m.storage_cap_new[i]
        m.storageCap = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = storageCap_rule)
        
        def maxStorageCap_rule(m,i):
            return m.Storage_cap_init[i] + m.storage_cap_new[i]  <= m.Storage_cap_max[i]
        m.maxStorageCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxStorageCap_rule)
        
        def hydrogenBalance_rule(m,t,i):
            return m.hydrogen_direct[t,i] + m.hydrogen_from_storage[t,i] \
                    + m.hydrogen_import[t,i] == m.Hydrogen_demand[t,i]
        m.hydrogenBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = hydrogenBalance_rule)
        
        
        def elecCap_rule(m,t,i):
            return m.H2_direct_eff*m.hydrogen_direct[t,i] \
                    + m.H2_storage_eff*m.hydrogen_to_storage[t,i] <=  m.Elec_cap_init[i] + m.elec_cap_new[i]
        m.elecCap = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = elecCap_rule)
        
        def maxElecCap_rule(m,i):
            return m.Elec_cap_init[i] + m.elec_cap_new[i]  <= m.Elec_cap_max[i]
        m.maxElecCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxElecCap_rule)
        
        # Energy balance
        def energyBalance_rule(m,t,i):
            if i in m.HYDROGEN_AT_NODE.keys():
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Consumer_load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum( m.H2_direct_eff*m.hydrogen_direct[t,j] \
                        + m.H2_storage_eff*m.hydrogen_to_storage[t,j]
                        for j in m.HYDROGEN_AT_NODE[i])
            else:
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Consumer_load[t,j] for j in m.LOAD_AT_NODE[i]) 

        m.energyBalance = pe.Constraint(m.TIME, m.NORMAL_NODES, rule = energyBalance_rule)
             
        # DC power flow
        def referenceNode_rule(m,t):
            return m.voltage_angle[t,m.NODES[1]] == 0.0
        m.ref_node = pe.Constraint(m.TIME, rule = referenceNode_rule)
        
        def branchFlow_rule(m,t,i,j):
            return m.branch_flow[t,i,j] == (1/m.Reactance[i,j])*(m.voltage_angle[t,i]-m.voltage_angle[t,j])
        m.branchFlow = pe.Constraint(m.TIME, m.BRANCHES, rule = branchFlow_rule)
        
        def branchFlowLimit_rule(m,t,i,j):
            if not np.isinf(m.Trans_cap[i,j]):
                return -m.Trans_cap[i,j]*(1/m.Ref_power)<= m.branch_flow[t,i,j] <= m.Trans_cap[i,j]*(1/m.Ref_power)
            else:
                return -10000*(1/m.Ref_power)<= m.branch_flow[t,i,j] <= 10000*(1/m.Ref_power)
        m.branchFlowLimit = pe.Constraint(m.TIME, m.BRANCHES, rule = branchFlowLimit_rule )
        
#        def expLimit_rule(m,t,i):
#            return m.exp[t,i] <= m.Exp_profile[t,i]
#        m.expLimit = pe.Constraint(m.TIME, m.MARKET_NODES, rule = expLimit_rule)
#        
#        def impLimit_rule(m,t,i):
#            return m.imp[t,i] <= m.Imp_profile[t,i]
#        m.impLimit = pe.Constraint(m.TIME, m.MARKET_NODES, rule = impLimit_rule)
        
        def nodalBalance_rule(m,t,i):
            return m.imp[t,i] - m.exp[t,i] == m.Ref_power*sum(m.Branch_dir_at_node[i,j]*m.branch_flow[t,j] for j in m.BRANCHES_AT_NODE[i])
        m.nodalBalance = pe.Constraint(m.TIME, m.NODES, rule = nodalBalance_rule)        
        
        def obj_rule(m):
            return sum(sum(m.External_price[t,i]*m.imp[t,i] - 1.01*m.External_price[t,i]*m.exp[t,i] for i in m.MARKET_NODES) \
                         -sum(m.Rationing_cost*m.rat[t,i] for i in m.NORMAL_NODES) \
                         -sum(m.Spill_cost*m.spill[t,i] for i in m.HYDRO_POWER_PLANTS) \
                         -sum(m.Hydrogen_import_cost*m.hydrogen_import[t,i] for i in m.HYDROGEN_PLANTS) for t in m.TIME) \
                    - sum(m.Drain_cost*m.drain_dev[i] + m.Fill_cost*m.fill_dev[i] for i in m.HYDRO_POWER_PLANTS)\
                    - m.Period_ratio*sum(m.P_cap_cost[i]*m.prod_cap_new[i] for i in m.WIND_POWER_PLANTS)\
                    - m.Period_ratio*sum(m.Storage_cap_cost[i]*m.storage_cap_new[i] + \
                         m.Elec_cap_cost[i]*m.elec_cap_new[i] for i in m.HYDROGEN_PLANTS)

        m.obj = pe.Objective(rule = obj_rule, sense = pe.maximize)
        
        return m

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def detData(obj):
    
    
    GW2MW = 1000
    KW2MW = 0.001
        
    di = {}
    ##Set##
    di['TIME'] = {None: list(obj.timerange)}
    #di['TIME'] = {None: list(range(3))}
    di['LAST_TIME'] = {None: [list(obj.timerange)[-1]]}
    
    node_data = obj.data.bus
    
    di['NODES'] = {None: node_data.index.tolist()}
    di['NORMAL_NODES'] = {None: node_data.loc[node_data.type == 'N'].index.tolist()}
    di['MARKET_NODES'] = {None: node_data.loc[node_data.type == 'M'].index.tolist()}
    
    line_data = obj.data.line
    
    branch_indx = []
    for i in line_data.index:
        branch_indx.append((line_data.from_bus[i],line_data.to_bus[i]))
    
    di['BRANCHES'] = {None: branch_indx }
    
    
    def getBranchesAtNode():
        out = {}
        for node in di['NODES'][None]:
            for i,j in di['BRANCHES'][None]:
                if i == node or j == node:
                    if node not in out.keys():
                        out[node] = []
                    out[node].append((i,j))
        return out
    
    di['BRANCHES_AT_NODE'] = getBranchesAtNode()


    hydro_data = obj.data.hydro_power
    
    di['HYDRO_POWER_PLANTS'] = {None: [ 'H'+str(i) for i in hydro_data.bus_indx.tolist()]}
    hydro_data['id'] = di['HYDRO_POWER_PLANTS'][None]
    
    wind_data = obj.data.wind_power
    
    di['WIND_POWER_PLANTS'] = {None: ['W'+str(i) for i in wind_data.bus_indx.tolist()]}
    wind_data['id'] = di['WIND_POWER_PLANTS'][None]
    
    di['POWER_PLANTS'] = {None: di['HYDRO_POWER_PLANTS'][None]+di['WIND_POWER_PLANTS'][None]}
    
    consumer_data = obj.data.consumer
    
    di['CONSUMERS'] = {None: ['C'+str(i) for i in consumer_data.bus_indx]}
    consumer_data['id'] = di['CONSUMERS'][None]
    
    hydrogen_data = obj.data.hydrogen
    
    di['HYDROGEN_PLANTS'] = {None: ['E'+str(i) for i in hydrogen_data.bus_indx]}
    hydrogen_data['id'] = di['HYDROGEN_PLANTS'][None]
    
          
    di['GEN_AT_NODE'] = {i:[j for j in di['POWER_PLANTS'][None]
                    if int(j[1:]) == i] for i in di['NORMAL_NODES'][None]}
    
    di['LOAD_AT_NODE'] = {i:[j for j in (di['CONSUMERS'][None] )#+di['HYDROGEN_PLANTS'][None])
                    if int(j[1:]) == i] for i in di['NORMAL_NODES'][None]}

    di['HYDROGEN_AT_NODE'] = {i:[j for j in (di['HYDROGEN_PLANTS'][None] )#+di['HYDROGEN_PLANTS'][None])
                    if int(j[1:]) == i] for i in di['NORMAL_NODES'][None]}
    
    ##Parameters##
    di['NTime'] = {None: len(obj.timerange)}
    di['Period_ratio'] = {None: len(obj.timerange)/8760}
    
    prices = obj.data.prices    
    
    ex_price = {}
    for i in di['MARKET_NODES'][None]:
        if node_data.loc[i,'market_area'] == np.NaN:
            print('Market node %d is missing a market area reference!' % int(i))
            continue
        else:
            price_series = prices[node_data.loc[i,'market_area']]
        for t in di['TIME'][None]:
            date = obj.time[t]
            ex_price[(t,i)] = float(price_series.loc[date])
    
    
    di['External_price'] = ex_price 
    
    param = obj.data.param
    
    consumer_load = {}
    for i in di['CONSUMERS'][None]:
        time = obj.time[t]
        for t in di['TIME'][None]:
            consumer_load[t,i] = obj.data.load.loc[time,i]
    di['Consumer_load'] = consumer_load

    
    di['H2_storage_eff'] = {None: float(param['storage_eff'].values[0]*KW2MW)} # MWh/Nm^3        
    di['H2_direct_eff'] = {None: float(param['direct_eff'].values[0]*KW2MW)} # MWh/Nm^3
    di['Hydrogen_import_cost'] = {None: float(param['import_cost'].values[0])} # €/Nm3

    storage_cap_max = {}
    storage_cap_init = {}
    storage_cap_cost = {}
    elec_cap_max = {}
    elec_cap_init = {}
    elec_cap_cost = {}
    hydrogen_demand = {}
    init_storage = {}
    for i in hydrogen_data.id:
        indx = hydrogen_data.index[(hydrogen_data.id == i)].values[0]
        storage_cap_max[i] = hydrogen_data.get_value(indx,'storage_cap_pot')
        storage_cap_init[i] = hydrogen_data.get_value(indx,'storage_cap')
        storage_cap_cost[i] = hydrogen_data.get_value(indx,'storage_cost')
        elec_cap_max[i] = hydrogen_data.get_value(indx,'elec_cap_pot')
        elec_cap_init[i] = hydrogen_data.get_value(indx,'elec_cap')
        elec_cap_cost[i] = hydrogen_data.get_value(indx,'elec_cost')
        init_storage[i] = hydrogen_data.get_value(indx,'init_storage')
        for t in di['TIME'][None]:
            hydrogen_demand[t,i] = float(obj.data.hydrogen_demand_series.get_value(t,i))
    di['Storage_cap_max'] = storage_cap_max
    di['Storage_cap_init'] = storage_cap_init
    di['Storage_cap_cost'] = storage_cap_cost
    di['Elec_cap_max'] = elec_cap_max
    di['Elec_cap_init'] = elec_cap_init
    di['Elec_cap_cost'] = elec_cap_cost
    di['Initial_storage'] = init_storage
    di['Hydrogen_demand'] = hydrogen_demand
    
    pmax = {}
    pinit = {}
    pcost = {}
    wind_profile = {}
            
    for i in wind_data.id:
        wind_indx = wind_data.index[wind_data.id == i].values[0]
        pinit[i] = float(wind_data.get_value(wind_indx,'p_max'))
        pmax[i] = float(wind_data.get_value(wind_indx,'pot_cap'))
        pcost[i] = float(wind_data.get_value(wind_indx,'inv_cost'))
        series_name = wind_data.get_value(wind_indx,'name')
        for t in di['TIME'][None]:
            time = obj.time[t]
            wind_profile[t,i] = float(obj.data.wind_power_production.loc[time, series_name])
        
    di['Wind_profile'] = wind_profile
    di['Fill_cost'] = {None: float(param.fill_cost.values[0])} #NOK/MWh
    di['Drain_cost'] = {None: float(param.drain_cost.values[0])} #NOK/MWh
    di['Spill_cost'] = {None: float(param.spill_cost.values[0])} #NOK/MWh
    
    inflow = {}
    inflow_ureg = {}
    res_max = {}
    initial_res = {}
    for i in hydro_data.id:
        hydro_indx = hydro_data.index[hydro_data.id == i].values[0]
        pmax[i] = float(hydro_data.get_value(hydro_indx,'p_max'))
        pinit[i] = pmax[i] # No new investments in hydro power
        pcost[i] = 0
        res_max[i] = float(hydro_data.get_value(hydro_indx,'res_max'))*GW2MW
        initial_res[i] = float(hydro_data.get_value(hydro_indx,'init_res'))
        for t in di['TIME'][None]:
            inflow[t,i] = float(obj.data.inflow.get_value(t,i))
            inflow_ureg[t,i] = float(obj.data.inflow_ureg.get_value(t,i))
        
    di['Inflow'] = inflow
    di['Inflow_ureg'] = inflow_ureg
    di['Res_cap'] = res_max
    di['Initial_res'] = initial_res
    
    di['P_cap_max'] = pmax
    di['P_cap_init'] = pinit
    di['P_cap_cost'] = pcost
        
    trans_cap = {}
    reactance = {}
    for i in line_data.index:
        from_bus = line_data.get_value(i,'from_bus')
        to_bus = line_data.get_value(i,'to_bus')
        cap = line_data.get_value(i,'cap')
        trans_cap[from_bus, to_bus] = float(cap)
        x = line_data.get_value(i,'reactance')
        reactance[from_bus, to_bus] = np.float(x)

    di['Trans_cap'] = trans_cap            
    di['Reactance'] = reactance
    di['Ref_power'] = {None: param['ref_power'].values[0]} # MW
    di['Rationing_cost'] = {None: param.get_value(0,'rat_cost')}
    
    def getBranchDirAtNode():
        out = {}
        for node in di['NODES'][None]:
            for i,j in di['BRANCHES_AT_NODE'][node]:
                if i == node:
                    out[node,i,j] = -1
                elif j == node:
                    out[node,i,j] = 1
        return out
    di['Branch_dir_at_node'] = getBranchDirAtNode()
    
    return {'detData':di}
