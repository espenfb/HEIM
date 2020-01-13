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
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class deterministicModel(object):
    ''' Deterministic investment model for regional power system with hydogen loads. '''
    
    def __init__(self, time_data, dirs, mutables = {}):
         
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
            
        self.mutables = mutables
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
        
        
    def buildModel(self):
        
        print('Building deterministic investment model...')
        self.detModel = buildDetModel(mutables = self.mutables)
    
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
        self.opt = pe.SolverFactory('gurobi', solver_io='python')#, verbose = True)
        #self.opt.options["Method"] = 2 #Chooses the interior point method 
        #opt.options["NodeMethod"] = 2 #Chooses the interior point method 
        #self.opt.options["Crossover"] = 0 #Turn off the crossover after the interior point method
        #self.opt.options["QCPDual"] = 0
    
        if printOutput:
                print('Solving deterministic operation model...')
            
        # Solve model
        start_time = time.time()
        self.pyomo_res = self.opt.solve(self.detModelInstance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True,
                        options={"Method": 2,
                                "Crossover": 0,
                                 'BarHomogeneous':1})#,
                          #       "QCPDual":0})#,
#                                 "NodeMethod": 2,
#                                 "MIPGap": 0.01,
#                                 "MIPFocus": 3)
    
        #self.detModelInstance.write('model.mps',
        #                        io_options={'symbolic_solver_labels':True})

        self.solution_time = time.time()-start_time
        
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
            
        pd.DataFrame.from_dict({'sol_time':[self.solution_time]}).to_csv(save_dir + 'sol_time.csv')

        dmr.saveDetRes(self,save_dir)
        
    def importResults(self, import_dir):
        ''' Importing results from files. '''
        
        dmr.importDetRes(self, import_dir)

def buildDetModel(mutables = {}):
        ''' Defines abstract investment model, defines sets, parameters, variables,
        constraints and objective function.'''
        
        mutable_dict = {'inv_cost': False,
                            'CO2_cost': False,
                            'H2_load_scaling': True}
        
        for i in mutables.keys():
            mutable_dict[i] = mutables[i]
        
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
        #m.BIOMASS_POWER_PLANTS = pe.Set()
        #m.COAL_POWER_PLANTS = pe.Set()
        #m.GAS_POWER_PLANTS = pe.Set()
        #m.NUCLEAR_POWER_PLANTS = pe.Set()
        m.SOLAR_POWER_PLANTS = pe.Set()
        m.WIND_POWER_PLANTS = pe.Set()  
        m.POWER_PLANTS = pe.Set()
        m.RENEWABLE_POWER_PLANTS = pe.Set()
        m.THERMAL_POWER_PLANTS = pe.Set()
        
        m.HYDROGEN_PLANTS = pe.Set()
        m.ELECTROLYSIS = pe.Set()
        m.H2_STORAGE = pe.Set()
        m.HYDROGEN_COMPONENTS = pe.Set()
        
        m.BATTERY_PLANTS = pe.Set()
        m.BATTERY_POWER = pe.Set()
        m.BATTERY_ENERGY = pe.Set()
        m.BATTERY_COMPONENTS = pe.Set()
        
        m.STORAGE_PLANTS = pe.Set()
        
        m.LOAD = pe.Set()
        m.H2_LOAD = pe.Set()
        
        m.GEN_AT_NODE = pe.Set(m.NODES)
        m.LOAD_AT_NODE = pe.Set(m.NODES)
        m.H2_LOAD_AT_NODE = pe.Set(m.NODES)
        m.H2PLANT_AT_NODE = pe.Set(m.NODES)
        m.BATTERY_AT_NODE = pe.Set(m.NODES)
        m.STORAGE_AT_NODE = pe.Set(m.NODES)
        
        m.COMPONENTS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        m.ELECTROLYSIS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        m.STORAGE_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
        
        m.COMPONENTS_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
        m.POWER_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
        m.ENERGY_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
        
        m.STORAGE_POWER_AT_PLANT = pe.Set(m.STORAGE_PLANTS)
        m.STORAGE_ENERGY_AT_PLANT = pe.Set(m.STORAGE_PLANTS)
        
        m.TYPE_TO_PLANTS = pe.Set(m.PLANT_TYPES)
        m.TYPE_TO_THERMAL_PLANTS = pe.Set(m.THERMAL_PLANT_TYPES)
        
        ##Parameters##
        m.NTime = pe.Param(within = pe.Integers)
        m.Period_ratio = pe.Param(within = pe.NonNegativeReals)
        
        m.Rationing_cost = pe.Param(within = pe.NonNegativeReals)
        m.CO2_cost = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['CO2_cost' ])
        
        m.Load = pe.Param(m.TIME, m.LOAD, within = pe.NonNegativeReals)
        m.H2_load = pe.Param(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.Emission_coef = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Inv_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, mutable = mutable_dict['inv_cost'])
        m.Fixed_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
        m.Var_cost = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Retirement_cost = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Ramp_rate = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Max_num_plants = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Plant_size = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Min_prod = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Init_cap = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.Solar_cap_pot = pe.Param(m.SOLAR_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_cap_inst = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_cap_pot = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_profile_inst = pe.Param(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)   
        m.Wind_profile_pot = pe.Param(m.TIME, m.WIND_POWER_PLANTS, within = pe.NonNegativeReals) 
        m.Solar_profile = pe.Param(m.TIME, m.SOLAR_POWER_PLANTS, within = pe.NonNegativeReals) 
        
        m.H2_load_scaling = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['H2_load_scaling'])
        
        m.H2_storage_eff = pe.Param(within = pe.NonNegativeReals)
        m.H2_direct_eff = pe.Param(within = pe.NonNegativeReals)
        m.Hydrogen_import_cost = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_import_cost_ccs = pe.Param(m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_CO2_emissions = pe.Param(within = pe.NonNegativeReals)
        m.Hydrogen_CO2_emissions_ccs = pe.Param(within = pe.NonNegativeReals)
        m.Initial_storage = pe.Param(m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        m.Hydrogen_demand = pe.Param(m.TIME, m.H2_LOAD, within = pe.NonNegativeReals)
        
        m.Battery_in_ratio = pe.Param(within = pe.NonNegativeReals)
        m.Battery_out_ratio = pe.Param(within = pe.NonNegativeReals)
        
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
        m.available_plants = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.retired_cap = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
#        m.available_cap = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.cur = pe.Var(m.TIME, m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals)
        m.gen_state = pe.Var(m.TIME, m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        
#        m.elec_cap_new = pe.Var(m.ELECTROLYSIS, within = pe.NonNegativeReals)
        m.hydrogen_direct = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import_ccs = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
#        m.storage_cap_new = pe.Var(m.H2_STORAGE, within = pe.NonNegativeReals)

        m.to_storage = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        m.from_storage = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        m.storage_level = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        
        m.rat = pe.Var(m.TIME, m.NODES, within = pe.NonNegativeReals)
        m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within = pe.Reals)
        m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
        m.new_branch_cap = pe.Var(m.NEW_BRANCHES, within = pe.Reals, bounds = (0,1))
        
        
        ## Constraints##
        
        # ALL POWER PLANTS
        
        def numPlants_rule(m,i):
            return m.Plant_size[i]*m.available_plants[i] <= m.Init_cap[i] + m.new_cap[i] - m.retired_cap[i]
        m.numPlants = pe.Constraint(m.THERMAL_POWER_PLANTS, rule = numPlants_rule)
        
        def genState_rule(m,t,i):
            return m.gen_state[t,i] <= m.available_plants[i]
        m.genState = pe.Constraint(m.TIME, m.THERMAL_POWER_PLANTS, rule = genState_rule)
        
        def maxPlant_rule(m,i):
            return m.available_plants[i] <= m.Max_num_plants[i]
        m.maxPlant = pe.Constraint(m.THERMAL_POWER_PLANTS, rule = maxPlant_rule)
        
        def maxProd_rule(m,t,i):
            return m.prod[t,i]  <= m.Plant_size[i]*m.gen_state[t,i]
        m.maxProd = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = maxProd_rule)
        
        def minProd_rule(m,t,i):
            return m.prod[t,i]  >= m.Min_prod[i]*m.gen_state[t,i]
        m.minProd = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = minProd_rule)
        
#        def startUp_rule(m,t,i):
#            return m.gen_state[t,i] == m.gen_state[t-1,i] + m.start_up[t,i] - m.shut_down[t,i] 
#        m.startUp = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = startUp_rule)
        
        def rampUpLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t,i] - m.prod[t-1,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.gen_state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampUpLimit = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = rampUpLimit_rule)
        
        def rampDownLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t-1,i] - m.prod[t,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.gen_state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampDownLimit = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = rampDownLimit_rule)
            
#        def uptime_rule(m,t,i):
#            
#            
#            
#        def downtime_rule(m,t,i):
        
        def maxSolarCap_rule(m,i):
            return m.new_cap[i]  <= m.Solar_cap_pot[i]
        m.maxSolarCap = pe.Constraint(m.SOLAR_POWER_PLANTS, rule = maxSolarCap_rule)
        
        def maxWindCap_rule(m,i):
            return m.new_cap[i]  <= m.Wind_cap_pot[i]
        m.maxWindCap = pe.Constraint(m.WIND_POWER_PLANTS, rule = maxWindCap_rule)
        
        # WIND POWER
        def windBalance_rule(m,t,i):
            if pe.value(m.Init_cap[i]) + pe.value(m.Wind_cap_pot[i]) > 0:
                return m.prod[t,i] + m.cur[t,i] == m.Wind_profile_inst[t,i]*m.Init_cap[i] \
                        +  m.Wind_profile_pot[t,i]*m.new_cap[i]
            else:
                return m.prod[t,i] + m.cur[t,i] == 0.0
        m.windBalance = pe.Constraint(m.TIME,m.WIND_POWER_PLANTS,
                                      rule = windBalance_rule)   
        # SOLAR POWER
        def solarBalance_rule(m,t,i):
            if pe.value(m.Solar_cap_pot[i]) > 0:
                return m.prod[t,i] + m.cur[t,i] == \
                        m.Solar_profile[t,i]*(m.Init_cap[i] + m.new_cap[i])
            else:
                return m.prod[t,i] + m.cur[t,i] == 0.0       
        m.solarBalance = pe.Constraint(m.TIME,m.SOLAR_POWER_PLANTS,
                                      rule = solarBalance_rule) 
        
#       Storage plants       
        def storageBalance_rule(m,t,i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            if t == 0:
                return m.storage_level[t,i] == m.Initial_storage[i]*(m.new_cap[j]) + m.to_storage[t,i] - m.from_storage[t,i]
            else:
                return m.storage_level[t,i] == m.storage_level[t-1,i] + m.to_storage[t,i] - m.from_storage[t,i]
        m.storageBalance = pe.Constraint(m.TIME, m.STORAGE_PLANTS, rule = storageBalance_rule)
        
        def endStorage_rule(m,t,i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            return m.storage_level[t,i] == m.Initial_storage[i]*(m.new_cap[j])
        m.endStorage = pe.Constraint(m.LAST_TIME, m.STORAGE_PLANTS, rule = endStorage_rule)
        
        def storageEnergyCap_rule(m, t, i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            return m.storage_level[t,i] <= m.new_cap[j]
        m.storageEnergyCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS, rule = storageEnergyCap_rule)
        
        def storageInPowerCap_rule(m, t, i):
            j = m.STORAGE_POWER_AT_PLANT[i]
            if i in m.BATTERY_PLANTS:
                return m.Battery_in_ratio*m.to_storage[t,i] <= m.new_cap[j]
            elif i in m.HYDROGEN_PLANTS:
                return m.hydrogen_direct[t,i] \
                    + m.to_storage[t,i] <= m.new_cap[j]
        m.storageInPowerCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS, rule = storageInPowerCap_rule)

        def storageOutPowerCap_rule(m, t, i):
            j = m.STORAGE_POWER_AT_PLANT[i]
            if i in m.BATTERY_PLANTS:
                return m.Battery_out_ratio*m.from_storage[t,i] <= m.new_cap[j]
            else:
                return pe.Constraint.Skip
        m.storageOutPowerCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS, rule = storageOutPowerCap_rule)
            
        
        
#        def maxStorageCap_rule(m,i):
#            return m.storage_cap_new[i]  <= m.Storage_cap_max[i]
#        m.maxStorageCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxStorageCap_rule)
        
        def hydrogenBalance_rule(m,t,i):
            return m.hydrogen_direct[t,i] + m.from_storage[t,i] \
                    + m.hydrogen_import[t,i] + m.hydrogen_import_ccs[t,i] \
                    == m.H2_load[t,i]*m.H2_load_scaling
        m.hydrogenBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = hydrogenBalance_rule)
        
        
#        def elecCap_rule(m,t,i):
#            j = m.ELECTROLYSIS_AT_H2PLANT[i]
#            return m.H2_direct_eff*m.hydrogen_direct[t,i] \
#                    + m.H2_storage_eff*m.hydrogen_to_storage[t,i] <= m.new_cap[j]
#        m.elecCap = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS, rule = elecCap_rule)
        
#        def maxElecCap_rule(m,i):
#            return m.elec_cap_new[i]  <= m.Elec_cap_max[i]
#        m.maxElecCap = pe.Constraint(m.HYDROGEN_PLANTS, rule = maxElecCap_rule)
        
        # Energy balance
        def energyBalance_rule(m,t,i):
            if (i in m.H2PLANT_AT_NODE.keys()) and (i in m.BATTERY_AT_NODE.keys()):
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
                        + m.H2_storage_eff*m.to_storage[t,j]
                        for j in m.H2PLANT_AT_NODE[i]) \
                        + sum(m.Battery_in_ratio*m.to_storage[t,j] \
                        - m.Battery_out_ratio*m.from_storage[t,j]
                        for j in m.BATTERY_AT_NODE[i])
            elif  (i in m.H2PLANT_AT_NODE.keys()):
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
                        + m.H2_storage_eff*m.to_storage[t,j]
                        for j in m.H2PLANT_AT_NODE[i])
            elif (i in m.BATTERY_AT_NODE.keys()):
                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum(m.Battery_in_ratio*m.to_storage[t,j] \
                        - m.Battery_out_ratio*m.from_storage[t,j]
                        for j in m.BATTERY_AT_NODE[i])
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
            return  m.Period_ratio*(sum(sum(m.Inv_cost[j]*m.new_cap[i]  for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)\
                    + sum(sum(m.Fixed_cost[j]*m.Plant_size[i]*m.available_plants[i] + m.Retirement_cost[j]*m.retired_cap[i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)\
                    + sum(m.Branch_cost[i]*m.Trans_cap[i]*m.new_branch_cap[i] for i in m.NEW_BRANCHES)) \
                    + sum(sum(sum((m.Var_cost[j] + m.Emission_coef[j]*m.CO2_cost)*m.prod[t,i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)
                          + sum(m.Rationing_cost*m.rat[t,i] for i in m.NODES) \
                          + sum((m.Hydrogen_import_cost[i] + m.Hydrogen_CO2_emissions*m.CO2_cost)*m.hydrogen_import[t,i]
                                + (m.Hydrogen_import_cost_ccs[i] + m.Hydrogen_CO2_emissions_ccs*m.CO2_cost)*m.hydrogen_import_ccs[t,i] for i in m.HYDROGEN_PLANTS) for t in m.TIME)

        m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
        
        return m



def detData(obj):
    ''' Data input for the investment model according to the abstract model.'''
    
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
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('PlantType', inplace = True)
    
    h2_plant_char = copy.copy(obj.data.hydrogen_plant_char)
    h2_plant_char.set_index('Type', inplace = True)
    
    di['PLANT_TYPES'] = {None: plant_char.index.to_list() + h2_plant_char.index.to_list()}
    thermal_plants = plant_char.index[obj.data.plant_char['Variable cost ($/MWh)'] > 0]
    di['THERMAL_PLANT_TYPES'] = {None: thermal_plants.to_list()}
    
    obj.type2prefix = {'Biomass' : 'B', 'CC Gas' : 'CCG', 'CT Gas' : 'CTG',
                            'ICE Gas' : 'ICEG', 'CCS Gas' : 'CCSG',
                            'Coal' : 'C', 'CCS Coal' : 'CCSC', 'Nuclear' : 'N',
                            'Solar' : 'S', 'Wind' : 'W', 'Elec' : 'E',
                            'H2_Storage' : 'HS', 'Hydrogen': 'H', 'Load': 'L',
                            'H2_Load': 'H2L', 'Battery': 'ES',
                            'Battery Power': 'ESP', 'Battery Energy':'ESE'}
    
    di['BIOMASS_POWER_PLANTS'] = {None: [obj.type2prefix['Biomass'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['COAL_POWER_PLANTS'] = {None: [obj.type2prefix['Coal'] +'%.2d' % i for i in installed.Bus.tolist()]}
    di['CCS_COAL_POWER_PLANTS'] = {None: [obj.type2prefix['CCS Coal'] +'%.2d' % i for i in installed.Bus.tolist()]}
    di['CC_GAS_POWER_PLANTS'] = {None: [obj.type2prefix['CC Gas'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['CT_GAS_POWER_PLANTS'] = {None: [obj.type2prefix['CT Gas'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['ICE_GAS_POWER_PLANTS'] = {None: [obj.type2prefix['ICE Gas'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['CCS_GAS_POWER_PLANTS'] = {None: [obj.type2prefix['CCS Gas'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['NUCLEAR_POWER_PLANTS'] = {None: [obj.type2prefix['Nuclear'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['SOLAR_POWER_PLANTS'] = {None: [obj.type2prefix['Solar'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['WIND_POWER_PLANTS'] = {None: [obj.type2prefix['Wind'] + '%.2d' % i for i in installed.Bus.tolist()]}
    
    solar_cap = copy.copy(obj.data.solar_cap)
    solar_cap.index = [ 'S%.2d' % i for i in obj.data.solar_cap.Bus.tolist()]
#    di['SOLAR_POWER_PLANTS'] = {None: solar_cap.index.tolist()}
#    
    wind_cap = obj.data.wind_cap
    #wind_cap = obj.data.wind_series.max(axis = 0).unstack()
    #wind_cap['Bus'] = [int(i) for i in wind_cap.index]
    
    for i in installed.Bus.tolist():
        if i not in wind_cap['Bus'].tolist():
            wind_cap.loc[i,'Inst_cap'] = 0.0
            wind_cap.loc[i,'Pot_cap'] = 0.0
            wind_cap.loc[i,'Bus'] = int(i)
    
    wind_cap.index = [ 'W%.2d' % i for i in wind_cap.Bus.tolist()]
    wind_cap.sort_index(inplace = True)
    wind_cap['Bus'] = [int(i) for i in wind_cap.Bus]
#    di['WIND_POWER_PLANTS'] = {None: wind_cap.index.tolist()}
    
    di['HYDROGEN_PLANTS'] = {None: [obj.type2prefix['Hydrogen'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['ELECTROLYSIS'] = {None: [obj.type2prefix['Elec'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['HYDROGEN_STORAGE'] = {None: [obj.type2prefix['H2_Storage'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['HYDROGEN_COMPONENTS'] = {None: di['ELECTROLYSIS'][None] \
                                  + di['HYDROGEN_STORAGE'][None]}
    
    di['BATTERY_PLANTS'] = {None: [obj.type2prefix['Battery'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['BATTERY_POWER'] = {None: [obj.type2prefix['Battery Power'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['BATTERY_ENERGY'] = {None: [obj.type2prefix['Battery Energy'] + '%.2d' % i for i in installed.Bus.tolist()]}
    di['BATTERY_COMPONENTS'] = {None: di['BATTERY_POWER'][None] \
                                  + di['BATTERY_ENERGY'][None]}
    
    di['STORAGE_PLANTS'] = {None: di['HYDROGEN_PLANTS'][None] + di['BATTERY_PLANTS'][None]}
    
    di['RENEWABLE_POWER_PLANTS'] = {None: di['WIND_POWER_PLANTS'][None] \
      + di['SOLAR_POWER_PLANTS'][None]}
    di['GAS_POWER_PLANTS'] = {None: di['CC_GAS_POWER_PLANTS'][None] \
      + di['CT_GAS_POWER_PLANTS'][None] + di['ICE_GAS_POWER_PLANTS'][None] 
      + di['CCS_GAS_POWER_PLANTS'][None]}
    di['THERMAL_POWER_PLANTS'] = {None: di['BIOMASS_POWER_PLANTS'][None] \
      + di['COAL_POWER_PLANTS'][None] + di['GAS_POWER_PLANTS'][None] \
      + di['NUCLEAR_POWER_PLANTS'][None] + di['CCS_COAL_POWER_PLANTS'][None]}
    di['POWER_PLANTS'] = {None: di['RENEWABLE_POWER_PLANTS'][None] \
      + di['THERMAL_POWER_PLANTS'][None]}
    
    load_series = copy.copy(obj.data.load_series)
    load_series.columns = [obj.type2prefix['Load'] + '%.2d' % int(i) for i in load_series.columns]
    load_series = load_series[load_series.index.isin(obj.time)]
    load_series.index = list(obj.timerange)
    
    
    di['LOAD'] = {None: load_series.columns.tolist()}
    
    hydrogen_load = copy.copy(obj.data.hydrogen_load)
    di['H2_LOAD'] = {None: [obj.type2prefix['H2_Load'] + '%.2d' % i for i in hydrogen_load.Bus.tolist()]}
#    di['HYDROGEN_PLANTS'] = {None: ['E%.2d' % i for i in hydrogen_load.Bus.tolist()]}
          
    di['GEN_AT_NODE'] = {i:[j for j in di['POWER_PLANTS'][None]
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['LOAD_AT_NODE'] = {i:[j for j in (di['LOAD'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['H2_LOAD_AT_NODE'] = {i:[j for j in (di['H2_LOAD'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}

    di['H2PLANT_AT_NODE'] = {i :[j for j in (di['HYDROGEN_PLANTS'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['BATTERY_AT_NODE'] = {i :[j for j in (di['BATTERY_PLANTS'][None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['STORAGE_AT_NODE'] = {**di['H2PLANT_AT_NODE'], **di['BATTERY_AT_NODE']}
    
    di['COMPONENTS_AT_H2PLANT'] = {obj.type2prefix['Hydrogen'] + '%.2d' % i :[j for j in (di['HYDROGEN_COMPONENTS'] [None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['COMPONENTS_AT_BATTERY'] = {obj.type2prefix['Battery'] + '%.2d' % i :[j for j in (di['BATTERY_COMPONENTS'] [None])
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    di['ELECTROLYSIS_AT_H2PLANT'] = {i: [j] for j in di['ELECTROLYSIS'][None] for i in di['HYDROGEN_PLANTS'][None] if int(j[-2:]) == int(i[-2:]) }
    
    di['STORAGE_AT_H2PLANT'] = {i: [j] for j in di['HYDROGEN_STORAGE'][None] for i in di['HYDROGEN_PLANTS'][None] if int(j[-2:]) == int(i[-2:])}

    di['POWER_AT_BATTERY'] = {i: [j] for j in di['BATTERY_POWER'][None] for i in di['BATTERY_PLANTS'][None] if int(j[-2:]) == int(i[-2:]) }
    
    di['ENERGY_AT_BATTERY'] = {i: [j] for j in di['BATTERY_ENERGY'][None] for i in di['BATTERY_PLANTS'][None] if int(j[-2:]) == int(i[-2:])}
    
    di['STORAGE_POWER_AT_PLANT'] = {**di['ELECTROLYSIS_AT_H2PLANT'], **di['POWER_AT_BATTERY']}
    di['STORAGE_ENERGY_AT_PLANT'] = {**di['STORAGE_AT_H2PLANT'], **di['ENERGY_AT_BATTERY']}


    di['TYPE_TO_PLANTS'] = {'Biomass' : di['BIOMASS_POWER_PLANTS'][None],
                            'CC Gas' : di['CC_GAS_POWER_PLANTS'][None],
                            'CT Gas' : di['CT_GAS_POWER_PLANTS'][None],
                            'ICE Gas' : di['ICE_GAS_POWER_PLANTS'][None],
                            'CCS Gas' : di['CCS_GAS_POWER_PLANTS'][None],
                            'Coal' : di['COAL_POWER_PLANTS'][None],
                            'CCS Coal' : di['CCS_COAL_POWER_PLANTS'][None],
                            'Nuclear' : di['NUCLEAR_POWER_PLANTS'][None],
                            'Solar' : di['SOLAR_POWER_PLANTS'][None],
                            'Wind' : di['WIND_POWER_PLANTS'][None],
                            'Elec' : di['ELECTROLYSIS'][None], 
                            'H2_Storage' : di['HYDROGEN_STORAGE'][None],
                            'Battery Power': di['BATTERY_POWER'][None],
                            'Battery Energy':di['BATTERY_ENERGY'][None]} 
    
    di['TYPE_TO_THERMAL_PLANTS'] = {'Biomass' : di['BIOMASS_POWER_PLANTS'][None],
                            'CC Gas' : di['CC_GAS_POWER_PLANTS'][None],
                            'CT Gas' : di['CT_GAS_POWER_PLANTS'][None],
                            'ICE Gas' : di['ICE_GAS_POWER_PLANTS'][None],
                            'CCS Gas' : di['CCS_GAS_POWER_PLANTS'][None],
                            'Coal' : di['COAL_POWER_PLANTS'][None],
                            'CCS Coal' : di['CCS_COAL_POWER_PLANTS'][None],
                            'Nuclear' : di['NUCLEAR_POWER_PLANTS'][None]}
    
    di['PLANTS'] = {None: di['POWER_PLANTS'][None]\
                          + di['HYDROGEN_COMPONENTS'][None]
                          + di['BATTERY_COMPONENTS'][None]}
    
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
    h2_load = h2_load[h2_load.index.isin(obj.time)]
    h2_load.index = np.arange(len(h2_load.index))
    di['H2_load'] = h2_load.stack().to_dict()

    installed.set_index('Bus', inplace = True)
    init_cap = installed
    init_cap.fillna(0, inplace = True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (obj.type2prefix[j],i) : init_cap[i,j] for i,j in init_cap.keys()}
    di['Init_cap'] = init_cap_dict
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('PlantType', inplace = True)
    
    ramp_rate = copy.copy(plant_char['Ramp Rate (%/h)'])
    rate = {}
    for t in di['TYPE_TO_THERMAL_PLANTS'].keys(): 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            rate[p] = ramp_rate.loc[t]
    di['Ramp_rate'] = rate # %/h
    
    min_limit = copy.copy(plant_char['Minimum Generation Limit (MW)'])
    m_lim = {}
    for t in di['TYPE_TO_THERMAL_PLANTS'].keys(): 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            m_lim[p] = min_limit.loc[t]
    di['Min_prod'] = m_lim # MW
    
    max_num_plants = copy.copy(obj.data.max_num_plants)
    max_num_plants.set_index('Type', inplace = True)
    max_plants = {}
    for t in max_num_plants.index: 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            max_plants[p] = max_num_plants.plants.loc[t]
    di['Max_num_plants'] = max_plants # MW
    
    plant_size = plant_char['Typical Plant Size (MW)']
    p_size = {}
    for t in di['TYPE_TO_THERMAL_PLANTS'].keys(): 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            p_size[p] = plant_size.loc[t]
    di['Plant_size'] = p_size # MW
    
    inv_cost = plant_char['Investment cost ($/MW-year)']
    inv_cost_dict = inv_cost.to_dict()
    h2_inv_cost = h2_plant_char['Capital cost [$/kg-year]']
    inv_cost_dict.update(h2_inv_cost)
    di['Inv_cost'] = inv_cost_dict
    
    fixed_cost = plant_char['Fixed cost ($/MW-year)']
    fixed_cost_dict = fixed_cost.to_dict()
    h2_fixed_cost = h2_plant_char['Fixed O&M [$/kg-year]']
    fixed_cost_dict.update(h2_fixed_cost)
    di['Fixed_cost'] = fixed_cost_dict
    
    var_cost = plant_char['Variable cost ($/MWh)']
    var_cost = var_cost[var_cost.index.isin(di['THERMAL_PLANT_TYPES'][None])]
    var_cost_dict = var_cost.to_dict()
#    h2_var_cost = h2_plant_char['Variable Costs [$/kg]']
#    var_cost_dict.update(h2_var_cost)
    di['Var_cost'] = var_cost_dict
    
    retirement_cost = copy.copy(obj.data.retirement_cost)
    retirement_cost.index = obj.data.retirement_cost.Type
    di['Retirement_cost'] = retirement_cost.Cost.to_dict()
    
    emission_coef = copy.copy(plant_char['Emission (kg/MWh)'])
    emission_coef = emission_coef[emission_coef.index.isin(di['THERMAL_PLANT_TYPES'][None])]
    di['Emission_coef'] = emission_coef.to_dict() # kg CO2/MWh
    
    di['Solar_cap_pot'] = solar_cap.Pot_cap.to_dict()
    
    solar_series = copy.copy(obj.data.solar_series)
    solar_series = solar_series[solar_series.index.isin(obj.time)]
    solar_series.index = pd.Index(np.arange(len(solar_series.index)))
    solar_series.rename(columns = {i : obj.type2prefix['Solar'] + '%.2d' % int(i) for i in solar_series.columns.tolist()},
                                  level = 0, inplace = True)
    di['Solar_profile'] = solar_series.round(4).stack(level = 0).to_dict()
    wind_cap.fillna(0, inplace = True)
    
    di['Wind_cap_inst'] = wind_cap.Inst_cap.to_dict()
    di['Wind_cap_pot'] = wind_cap.Pot_cap.to_dict()
    
    di['H2_storage_eff'] = {None: h2_plant_char.loc['Elec','Energy rate [MWh/kg]'] + 
                              h2_plant_char.loc['H2_Storage','Energy rate [MWh/kg]']} # MWh/Nm^3        
    di['H2_direct_eff'] = {None: h2_plant_char.loc['Elec','Energy rate [MWh/kg]']} # MWh/Nm^3
#    di['Hydrogen_import_cost'] = {None: float(param.import_cost.values[0])} # €/kg
#    di['Hydrogen_import_cost_ccs'] = {None: float(param.import_cost_ccs.values[0])} # €/kg
    di['Hydrogen_CO2_emissions'] = {None: float(param.CO2_H2_imp.values[0])} # kg/kg
    di['Hydrogen_CO2_emissions_ccs'] = {None: float(param.CO2_H2_imp_ccs.values[0])} # kg/kg
    di['Initial_storage'] = {i: 0.5 for i in di['STORAGE_PLANTS'][None]}
    
    hydrogen_ng = copy.copy(obj.data.hydrogen_ng)
    hydrogen_ng.set_index('Plant', inplace = True)
    di['Hydrogen_import_cost'] = hydrogen_ng.H2_ng.to_dict()
    di['Hydrogen_import_cost_ccs'] = hydrogen_ng.H2_ng_ccs.to_dict()
    
    di['H2_load_scaling'] = {None: 0.0}
    
    di['Battery_in_ratio'] = {None: float(param.battery_in_ratio.values[0])}
    di['Battery_out_ratio'] = {None: float(param.battery_out_ratio.values[0])} 
    
    wind_series = copy.copy(obj.data.wind_series)
    wind_series = wind_series[wind_series.index.isin(obj.time)]
    wind_series.index = np.arange(len(wind_series.index))
    wind_series.rename(columns = {i : obj.type2prefix['Wind'] + '%.2d' % int(i) for i in wind_series.columns.levels[0].tolist()},
                                  level = 0, inplace = True)
    
    di['Wind_profile_inst'] = wind_series.round(4).stack(level = 0).Inst_cap.fillna(0).to_dict()
    di['Wind_profile_pot'] = wind_series.round(4).stack(level = 0).Pot_cap.to_dict()
    
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
