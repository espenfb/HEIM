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
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import copy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class deterministicModel(object):
    ''' Deterministic model for regional power system with hydogen loads,
    wind power and hydro power. '''
    
    def __init__(self, time_data, dirs):
         
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        # Import system data
        self.data = sd.systemData(dirs)
        wind = self.data.wind_series
        solar = self.data.solar_series
             
        self.time = pd.date_range(start = self.start_date,
                                  end = self.end_date - relativedelta(hour = 1),
                                  freq = 'H')
        
        self.time = self.time[self.time.isin(self.data.load_series.index)]
        self.time = self.time[self.time.isin(wind.index)]
        self.time = self.time[self.time.isin(solar.index)]
        
        self.timerange = range(len(self.time))
        
        self.buildModel()
        
        
    def buildModel(self):
        
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
        
#    def printRes(self):
#        
#        print('Wind power capacity: ')
#        for i in self.detModelInstance.WIND_POWER_PLANTS:
#            print(i,': ', self.detModelInstance.P_cap_init[i],' + ', '%.2f' %
#                  self.detModelInstance.prod_cap_new[i].value)
#        
#        print()
#        
#        print('Electrolyser capacity: ')
#        for i in self.detModelInstance.HYDROGEN_PLANTS:
#            print(i,': ', self.detModelInstance.Elec_cap_init[i],' + ', '%.2f' %
#                  self.detModelInstance.elec_cap_new[i].value)
#        
#        print()
#            
#        print('Hydrogen storage capacity: ')
#        for i in self.detModelInstance.HYDROGEN_PLANTS:
#            print(i,': ', self.detModelInstance.Storage_cap_init[i],' + ', '%.2f' %
#                  self.detModelInstance.storage_cap_new[i].value)
            
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
        m.CURRENT_BRANCHES = pe.Set(dimen = 3)
        m.NEW_BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES_AT_NODE = pe.Set(m.NODES, dimen = 3)
#        m.NODES_AT_BRANCH = pe.Set(m.BRANCHES)
        
        m.PLANT_TYPES = pe.Set()
        m.THERMAL_PLANT_TYPES = pe.Set()
        m.PLANTS = pe.Set()
        m.BIOMASS_POWER_PLANTS = pe.Set()
        m.COAL_POWER_PLANTS = pe.Set()
        m.GAS_POWER_PLANTS = pe.Set()
        m.NUCLEAR_POWER_PLANTS = pe.Set()
        m.SOLAR_POWER_PLANTS = pe.Set()
        m.WIND_POWER_PLANTS = pe.Set()  
        m.POWER_PLANTS = pe.Set()
        m.RENEWABLE_POWER_PLANTS = pe.Set()
        m.THERMAL_POWER_PLANTS = pe.Set()
        
        m.HYDROGEN_PLANTS = pe.Set()
        m.ELECTROLYSIS = pe.Set()
        m.H2_STORAGE = pe.Set()
        m.HYDROGEN_COMPONENTS = pe.Set()
        
        m.LOAD = pe.Set()
        m.H2_LOAD = pe.Set()
        
        m.GEN_AT_NODE = pe.Set(m.NODES)
        m.LOAD_AT_NODE = pe.Set(m.NODES)
        m.H2_LOAD_AT_NODE = pe.Set(m.NODES)
        m.H2PLANT_AT_NODE = pe.Set(m.NODES)
        m.COMPONENTS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        m.ELECTROLYSIS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        m.STORAGE_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        m.TYPE_TO_PLANTS = pe.Set(m.PLANT_TYPES)
        m.TYPE_TO_THERMAL_PLANTS = pe.Set(m.THERMAL_PLANT_TYPES)
        
        ##Parameters##
        m.NTime = pe.Param(within = pe.Integers)
        m.Period_ratio = pe.Param(within = pe.NonNegativeReals)
        
        m.Rationing_cost = pe.Param(within = pe.NonNegativeReals)
        m.CO2_cost = pe.Param(within = pe.NonNegativeReals)
        
        m.Load = pe.Param(m.TIME, m.LOAD, within = pe.NonNegativeReals)
        m.H2_load = pe.Param(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.Fuel_cost = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Emission_coef = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Inv_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
        m.Init_cap = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.Solar_cap_pot = pe.Param(m.SOLAR_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_cap_inst = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_cap_pot = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_profile_inst = pe.Param(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)   
        m.Wind_profile_pot = pe.Param(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals) 
        m.Solar_profile_pot = pe.Param(m.TIME, m.SOLAR_POWER_PLANTS, within = pe.NonNegativeReals) 
        
        
        m.H2_storage_eff = pe.Param(within = pe.NonNegativeReals)
        m.H2_direct_eff = pe.Param(within = pe.NonNegativeReals)
        m.Hydrogen_import_cost = pe.Param(within = pe.NonNegativeReals)
        m.Hydrogen_CO2_emissions = pe.Param(within = pe.NonNegativeReals)
        m.Initial_storage = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_demand = pe.Param(m.TIME, m.H2_LOAD, within = pe.NonNegativeReals)
        
        m.Trans_cap = pe.Param(m.BRANCHES,within = pe.NonNegativeReals)
        m.Branch_cost = pe.Param(m.NEW_BRANCHES,within = pe.NonNegativeReals)
        m.Susceptance = pe.Param(m.BRANCHES,within = pe.Reals) # Non-Negative?
        m.Ref_power = pe.Param(within = pe.NonNegativeReals)
        m.Branch_dir_at_node = pe.Param(m.NODES,m.BRANCHES, within = pe.Integers)
                
        # Variables
        m.exp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        m.imp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        
        m.prod = pe.Var(m.TIME, m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.new_cap = pe.Var(m.PLANTS, within = pe.NonNegativeReals)
        m.cur = pe.Var(m.TIME, m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals)
        
#        m.elec_cap_new = pe.Var(m.ELECTROLYSIS, within = pe.NonNegativeReals)
        m.hydrogen_direct = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_to_storage = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_from_storage = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
#        m.storage_cap_new = pe.Var(m.H2_STORAGE, within = pe.NonNegativeReals)
        m.storage_level = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.rat = pe.Var(m.TIME, m.NODES, within = pe.NonNegativeReals)
        m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within = pe.Reals)
        m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
        m.new_branch_cap = pe.Var(m.NEW_BRANCHES, within = pe.Reals, bounds = (0,1))
        
        
        ## Constraints##
        
        # ALL POWER PLANTS
        def maxProd_rule(m,t,i):
            return m.prod[t,i]  <= m.Init_cap[i] + m.new_cap[i]
        m.maxProd = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = maxProd_rule)
        
        def maxSolarCap_rule(m,i):
            return m.new_cap[i]  <= m.Solar_cap_pot[i]
        m.maxSolarCap = pe.Constraint(m.SOLAR_POWER_PLANTS, rule = maxSolarCap_rule)
        
        def maxWindCap_rule(m,i):
            return m.new_cap[i]  <= m.Wind_cap_pot[i]
        m.maxWindCap = pe.Constraint(m.WIND_POWER_PLANTS, rule = maxWindCap_rule)
        
        # WIND POWER
        def windBalance_rule(m,t,i):
            if pe.value(m.Init_cap[i]) + pe.value(m.Wind_cap_pot[i]) > 0:
                return m.prod[t,i] + m.cur[t,i] == m.Wind_profile_inst[t,i] \
                        +  (m.new_cap[i]/m.Wind_cap_pot[i])*m.Wind_profile_pot[t,i]
            else:
                return m.prod[t,i] + m.cur[t,i] == 0.0
        m.windBalance = pe.Constraint(m.TIME,m.WIND_POWER_PLANTS,
                                      rule = windBalance_rule)   
        # SOLAR POWER
        def solarBalance_rule(m,t,i):
            if pe.value(m.Solar_cap_pot[i]) > 0:
                return m.prod[t,i] + m.cur[t,i] == \
                        (m.new_cap[i]/m.Solar_cap_pot[i])*m.Solar_profile_pot[t,i]
            else:
                return m.prod[t,i] + m.cur[t,i] == 0.0       
        m.solarBalance = pe.Constraint(m.TIME,m.SOLAR_POWER_PLANTS,
                                      rule = solarBalance_rule) 
        
#         Hydrogen plants       
        def storageBalance_rule(m,t,i):
            j = m.STORAGE_AT_H2PLANT[i]
            if t == 0:
                return m.storage_level[t,i] == m.Initial_storage[i]*(m.new_cap[j]) + m.hydrogen_to_storage[t,i] - m.hydrogen_from_storage[t,i]
            else:
                return m.storage_level[t,i] == m.storage_level[t-1,i] + m.hydrogen_to_storage[t,i] - m.hydrogen_from_storage[t,i]
        m.storageBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = storageBalance_rule)
        
        def endStorage_rule(m,t,i):
            j = m.STORAGE_AT_H2PLANT[i]
            return m.storage_level[t,i] == m.Initial_storage[i]*(m.new_cap[j])
        m.endStorage = pe.Constraint(m.LAST_TIME, m.HYDROGEN_PLANTS, rule = endStorage_rule)
        
        def storageCap_rule(m, t, i):
            j = m.STORAGE_AT_H2PLANT[i]
            return m.storage_level[t,i] <= m.new_cap[j]
        m.storageCap = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = storageCap_rule)
        
#        def maxStorageCap_rule(m,i):
#            return m.storage_cap_new[i]  <= m.Storage_cap_max[i]
#        m.maxStorageCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxStorageCap_rule)
        
        def hydrogenBalance_rule(m,t,i):
            return m.hydrogen_direct[t,i] + m.hydrogen_from_storage[t,i] \
                    + m.hydrogen_import[t,i] == m.H2_load[t,i]
        m.hydrogenBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = hydrogenBalance_rule)
        
        
        def elecCap_rule(m,t,i):
            j = m.ELECTROLYSIS_AT_H2PLANT[i]
            return m.H2_direct_eff*m.hydrogen_direct[t,i] \
                    + m.H2_storage_eff*m.hydrogen_to_storage[t,i] <= m.new_cap[j]
        m.elecCap = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = elecCap_rule)
        
#        def maxElecCap_rule(m,i):
#            return m.elec_cap_new[i]  <= m.Elec_cap_max[i]
#        m.maxElecCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxElecCap_rule)
        
        # Energy balance
        def energyBalance_rule(m,t,i):
            if i in m.H2PLANT_AT_NODE.keys():
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
                        + m.H2_storage_eff*m.hydrogen_to_storage[t,j]
                        for j in m.H2PLANT_AT_NODE[i])
            else:
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Consumer_load[t,j] for j in m.LOAD_AT_NODE[i]) 

        m.energyBalance = pe.Constraint(m.TIME, m.NODES, rule = energyBalance_rule)
             
        # DC power flow
        def referenceNode_rule(m,t):
            return m.voltage_angle[t,m.NODES[1]] == 0.0
        m.ref_node = pe.Constraint(m.TIME, rule = referenceNode_rule)
        
        def branchFlow_rule(m,t,n,i,j):
            return m.branch_flow[t,n,i,j] == m.Susceptance[n,i,j]*(m.voltage_angle[t,i]-m.voltage_angle[t,j])
        m.branchFlow = pe.Constraint(m.TIME, m.BRANCHES, rule = branchFlow_rule)
        
        def branchFlowLimit_rule(m,t,n,i,j):
            if not np.isinf(m.Trans_cap[n,i,j]):
                return (-m.Trans_cap[n,i,j], m.branch_flow[t,n,i,j], m.Trans_cap[n,i,j])
            else:
                return (-10000, m.branch_flow[t,n,i,j], 10000)
        m.branchFlowLimit = pe.Constraint(m.TIME, m.CURRENT_BRANCHES, rule = branchFlowLimit_rule )
        
        def newBranchFlowUpperLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Trans_cap[n, i,j])):
                return m.branch_flow[t,n,i,j] <= m.new_branch_cap[n,i,j]*m.Trans_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] <= 10000
        m.newBranchFlowUpperLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES, rule = newBranchFlowUpperLimit_rule )
        
        def newBranchFlowLowerLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Trans_cap[n, i,j])):
                return m.branch_flow[t,n,i,j] >= -m.new_branch_cap[n,i,j]*m.Trans_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] >= -10000
        m.newBranchFlowLowerLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES, rule = newBranchFlowLowerLimit_rule )
        
#        def maxTransCap_rule(m,n,i,j):
#            return m.new_branch_cap[n,i,j] <= m.Trans_cap[n,i,j] 
#        m.maxTransCap = pe.Constraint(m.NEW_BRANCHES, rule = maxTransCap_rule)
        
        def nodalBalance_rule(m,t,i):
            return m.imp[t,i] - m.exp[t,i] == m.Ref_power*sum(m.Branch_dir_at_node[i,j]*m.branch_flow[t,j] for j in m.BRANCHES_AT_NODE[i])
        m.nodalBalance = pe.Constraint(m.TIME, m.NODES, rule = nodalBalance_rule)        
        
        def obj_rule(m):
            return  m.Period_ratio*sum(sum(m.Inv_cost[j]*m.new_cap[i] for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)\
                    + m.Period_ratio*sum(m.Branch_cost[i]*m.Trans_cap[i]*m.new_branch_cap[i] for i in m.NEW_BRANCHES) \
                    + sum(sum(sum((m.Fuel_cost[j] + m.Emission_coef[j]*m.CO2_cost)*m.prod[t,i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)
                          + sum(m.Rationing_cost*m.rat[t,i] for i in m.NODES) \
                          + sum((m.Hydrogen_import_cost + m.Hydrogen_CO2_emissions*m.CO2_cost)*m.hydrogen_import[t,i] for i in m.HYDROGEN_PLANTS) for t in m.TIME)

        m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
        
        return m



def detData(obj):
    
    
    GW2MW = 1000
    KW2MW = 0.001
        
    di = {}
    ##Set##
    di['TIME'] = {None: list(obj.timerange)}
    #di['TIME'] = {None: list(range(3))}
    di['LAST_TIME'] = {None: [list(obj.timerange)[-1]]}
    node_data = copy.copy(obj.data.bus)
    
    di['NODES'] = {None: node_data.Bus.tolist()}

    line_data = copy.copy(obj.data.line)
    
    branch_indx = []
    new_branch_indx = []
    for i in line_data.index:
        if line_data.Type[i] == 'Existing':
            branch_indx.append((i,line_data.From[i],line_data.To[i]))
        elif line_data.Type[i] == 'New':
            new_branch_indx.append((i,line_data.From[i],line_data.To[i]))
        
    di['CURRENT_BRANCHES'] = {None: branch_indx }
    di['NEW_BRANCHES'] = {None: new_branch_indx }
    di['BRANCHES'] =  {None: di['CURRENT_BRANCHES'][None] + di['NEW_BRANCHES'][None]}

    def getBranchesAtNode():
        out = {}
        for node in di['NODES'][None]:
            for n,i,j in di['BRANCHES'][None]:
                if i == node or j == node:
                    if node not in out.keys():
                        out[node] = []
                    out[node].append((n,i,j))
        return out
    
    di['BRANCHES_AT_NODE'] = getBranchesAtNode()


    installed = copy.copy(obj.data.installed)
    
    di['PLANT_TYPES'] = {None: obj.data.inv_cost.Type}
    di['THERMAL_PLANT_TYPES'] = {None: obj.data.fuel_cost.Type}
    
    di['BIOMASS_POWER_PLANTS'] = {None: ['B%.2d' % i for i in installed.Bus.tolist()]}
    di['COAL_POWER_PLANTS'] = {None: ['C%.2d' % i for i in installed.Bus.tolist()]}
    di['GAS_POWER_PLANTS'] = {None: ['G%.2d' % i for i in installed.Bus.tolist()]}
    di['NUCLEAR_POWER_PLANTS'] = {None: ['N%.2d' % i for i in installed.Bus.tolist()]}
    di['SOLAR_POWER_PLANTS'] = {None: ['S%.2d' % i for i in installed.Bus.tolist()]}
    di['WIND_POWER_PLANTS'] = {None: ['W%.2d' % i for i in installed.Bus.tolist()]}
    
    solar_cap = copy.copy(obj.data.solar_cap)
    solar_cap.index = [ 'S%.2d' % i for i in obj.data.solar_cap.Bus.tolist()]
#    di['SOLAR_POWER_PLANTS'] = {None: solar_cap.index.tolist()}
#    
    #wind_cap = obj.data.wind_cap
    wind_cap = obj.data.wind_series.max(axis = 0).unstack()
    wind_cap['Bus'] = [int(i) for i in wind_cap.index]
    
    for i in installed.Bus.tolist():
        if i not in wind_cap['Bus'].tolist():
            wind_cap.loc[i,'Inst_cap'] = 0.0
            wind_cap.loc[i,'Pot_cap'] = 0.0
            wind_cap.loc[i,'Bus'] = int(i)
    
    wind_cap.index = [ 'W%.2d' % i for i in wind_cap.Bus.tolist()]
    wind_cap.sort_index(inplace = True)
    wind_cap['Bus'] = [int(i) for i in wind_cap.Bus]
#    di['WIND_POWER_PLANTS'] = {None: wind_cap.index.tolist()}
    
    di['HYDROGEN_PLANTS'] = {None: ['H%.2d' % i for i in installed.Bus.tolist()]}
    di['ELECTROLYSIS'] = {None: ['E%.2d' % i for i in installed.Bus.tolist()]}
    di['HYDROGEN_STORAGE'] = {None: ['HS%.2d' % i for i in installed.Bus.tolist()]}
    di['HYDROGEN_COMPONENTS'] = {None: di['ELECTROLYSIS'][None] \
                                  + di['HYDROGEN_STORAGE'][None]}
    
    di['RENEWABLE_POWER_PLANTS'] = {None: di['WIND_POWER_PLANTS'][None] \
      + di['SOLAR_POWER_PLANTS'][None]}
    di['THERMAL_POWER_PLANTS'] = {None: di['BIOMASS_POWER_PLANTS'][None] \
      + di['COAL_POWER_PLANTS'][None] + di['GAS_POWER_PLANTS'][None] \
      + di['NUCLEAR_POWER_PLANTS'][None]}
    di['POWER_PLANTS'] = {None: di['RENEWABLE_POWER_PLANTS'][None] \
      + di['THERMAL_POWER_PLANTS'][None]}
    
    load_series = copy.copy(obj.data.load_series)
    load_series.columns = ['L%.2d' % int(i) for i in load_series.columns]
    load_series = load_series[load_series.index.isin(obj.time)]#*1.15
    load_series.index = list(obj.timerange)
    
    
    di['LOAD'] = {None: load_series.columns.tolist()}
    
    hydrogen_load = copy.copy(obj.data.hydrogen_load)
    di['H2_LOAD'] = {None: ['H2L%.2d' % i for i in hydrogen_load.Bus.tolist()]}
#    di['HYDROGEN_PLANTS'] = {None: ['E%.2d' % i for i in hydrogen_load.Bus.tolist()]}
          
    di['GEN_AT_NODE'] = {i:[j for j in di['POWER_PLANTS'][None]
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['LOAD_AT_NODE'] = {i:[j for j in (di['LOAD'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['H2_LOAD_AT_NODE'] = {i:[j for j in (di['H2_LOAD'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}

    di['H2PLANT_AT_NODE'] = {i :[j for j in (di['HYDROGEN_PLANTS'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['COMPONENTS_AT_H2PLANT'] = {'H%.2d' % i :[j for j in (di['HYDROGEN_COMPONENTS'] [None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['ELECTROLYSIS_AT_H2PLANT'] = {i: [j] for j in di['ELECTROLYSIS'][None] for i in di['HYDROGEN_PLANTS'][None] if int(j[-2:]) == int(i[-2:]) }
    
    di['STORAGE_AT_H2PLANT'] = {i: [j] for j in di['HYDROGEN_STORAGE'][None] for i in di['HYDROGEN_PLANTS'][None] if int(j[-2:]) == int(i[-2:])}

    di['TYPE_TO_PLANTS'] = {'Biomass' : di['BIOMASS_POWER_PLANTS'][None],
                            'Gas' : di['GAS_POWER_PLANTS'][None],
                            'Coal' : di['COAL_POWER_PLANTS'][None],
                            'Nuclear' : di['NUCLEAR_POWER_PLANTS'][None],
                            'Solar' : di['SOLAR_POWER_PLANTS'][None],
                            'Wind' : di['WIND_POWER_PLANTS'][None],
                            'Elec' : di['ELECTROLYSIS'][None], 
                            'H2_Storage' : di['HYDROGEN_STORAGE'][None]} 
    
    di['TYPE_TO_THERMAL_PLANTS'] = {'Biomass' : di['BIOMASS_POWER_PLANTS'][None],
                            'Gas' : di['GAS_POWER_PLANTS'][None],
                            'Coal' : di['COAL_POWER_PLANTS'][None],
                            'Nuclear' : di['NUCLEAR_POWER_PLANTS'][None]}
    
    di['PLANTS'] = {None: di['POWER_PLANTS'][None]\
                          + di['HYDROGEN_COMPONENTS'][None]}
    
    ##Parameters##
    di['NTime'] = {None: len(obj.timerange)}
    di['Period_ratio'] = {None: len(obj.timerange)/8760}
    
    param = copy.copy(obj.data.parameters)
    
    di['Load'] = load_series.stack().to_dict()
    
    h2_load = pd.DataFrame(index = obj.data.load_series.index,
                           columns = di['HYDROGEN_PLANTS'][None])
    for i in di['HYDROGEN_PLANTS'][None]:
        indx = obj.data.hydrogen_load.Bus == int(i[-2:])
        value = obj.data.hydrogen_load.loc[indx,'high'].values/24 #kg/h
        if len(value) > 0:
            h2_load.loc[:,i] = value[0]    
    h2_load.fillna(0, inplace = True)
    h2_load = h2_load[h2_load.index.isin(obj.time)]/0.0899
    h2_load.index = np.arange(len(h2_load.index))
    di['H2_load'] = h2_load.stack().to_dict()

    installed.set_index('Bus', inplace = True)
    init_cap = pd.concat([installed,
                          wind_cap.set_index('Bus').Inst_cap.rename('Wind'),
                          solar_cap.set_index('Bus').Inst_cap.rename('Solar')],
                        axis = 1)
    init_cap.fillna(0, inplace = True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (j[0],i) : init_cap[i,j] for i,j in init_cap.keys()}
    di['Init_cap'] = init_cap_dict
    
    inv_cost = copy.copy(obj.data.inv_cost)
    inv_cost.index = obj.data.inv_cost.Type
    di['Inv_cost'] = inv_cost.Cost.to_dict()
    
    fuel_cost = copy.copy(obj.data.fuel_cost)
    fuel_cost.index = obj.data.fuel_cost.Type
    di['Fuel_cost'] = fuel_cost.Cost.to_dict()
    
    emission_coef = copy.copy(obj.data.emission)
    emission_coef.index = obj.data.emission.Type
    di['Emission_coef'] = emission_coef.Emission.to_dict() # kg CO2/MWh
    
    di['Solar_cap_pot'] = solar_cap.Pot_cap.to_dict()
    
    solar_series = copy.copy(obj.data.solar_series)
    solar_series = solar_series[solar_series.index.isin(obj.time)]
    solar_series.index = pd.Index(np.arange(len(solar_series.index)))
    solar_series.rename(columns = {i : 'S%.2d' % int(i) for i in solar_series.columns.tolist()},
                                  level = 0, inplace = True)
    di['Solar_profile_pot'] = solar_series.stack(level = 0).to_dict()
    wind_cap.fillna(0, inplace = True)
    
    di['Wind_cap_inst'] = wind_cap.Inst_cap.to_dict()
    di['Wind_cap_pot'] = wind_cap.Pot_cap.to_dict()
    
    di['H2_storage_eff'] = {None: float(param.storage_eff.values[0]*KW2MW)} # MWh/Nm^3        
    di['H2_direct_eff'] = {None: float(param.direct_eff.values[0]*KW2MW)} # MWh/Nm^3
    di['Hydrogen_import_cost'] = {None: float(param.import_cost.values[0])} # €/Nm3
    di['Hydrogen_CO2_emissions'] = {None: float(param.CO2_H2_imp.values[0])} # kg/Nm^3
    di['Initial_storage'] = {i: 0.5 for i in di['HYDROGEN_PLANTS'][None]} 
    
    wind_series = copy.copy(obj.data.wind_series)
    wind_series = wind_series[wind_series.index.isin(obj.time)]
    wind_series.index = np.arange(len(wind_series.index))
    wind_series.rename(columns = {i : 'W%.2d' % int(i) for i in wind_series.columns.levels[0].tolist()},
                                  level = 0, inplace = True)
    
    di['Wind_profile_inst'] = wind_series.stack(level = 0).Inst_cap.fillna(0).to_dict()
    di['Wind_profile_pot'] = wind_series.stack(level = 0).Pot_cap.to_dict()
    
    line_data.index = list(zip(line_data.index, line_data.From,line_data.To))
    di['Branch_cost'] = line_data[line_data.Type == 'New'].Cost.to_dict()
    di['Trans_cap'] = line_data.Cap.to_dict()
    di['Susceptance'] = line_data.B.to_dict()

    di['Ref_power'] = {None: param['ref_power'].values[0]} # MW
    di['Rationing_cost'] = {None: param.at[0,'rat_cost']}
    di['CO2_cost'] = {None: param.at[0,'CO2_cost']}
    

    def getBranchDirAtNode():
        out = {}
        for node in di['NODES'][None]:
            for n,i,j in di['BRANCHES_AT_NODE'][node]:
                if i == node:
                    out[node,n,i,j] = -1
                elif j == node:
                    out[node,n,i,j] = 1
        return out
    di['Branch_dir_at_node'] = getBranchDirAtNode()
    
    return {'detData':di}
