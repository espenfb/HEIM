# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:07:41 2017

@author: espenfb
"""

import pyomo.environ as pe
import os
import detModelRes as dmr
import systemData as sd
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import copy
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

GW2MW = 1000

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
        self.instance = self.detModel.create_instance(
                                data= self.detDataInstance,
                                name="Deterministic operation model",
                                namespace='detData')
        
        # Enable access to duals
        self.instance.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        
    def solve(self, printOutput = True):
        
        # Connect to solver
        self.opt = pe.SolverFactory('gurobi', solver_io='python')#, verbose = True)
    
        if printOutput:
                print('Solving deterministic operation model...')
            
        # Solve model
        start_time = time.time()
        self.pyomo_res = self.opt.solve(self.instance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True,
                        warmstart=True,
                        options={"Method": 2,
                                "Crossover": 0,
                                 'BarHomogeneous':1})#,
                          #       "QCPDual":0})#,
#                                 "NodeMethod": 2,
#                                 "MIPGap": 0.01,
#                                 "MIPFocus": 3)
    
        #self.instance.write('model.mps',
        #                        io_options={'symbolic_solver_labels':True})

        self.solution_time = time.time()-start_time
        
    def printModel(self, name = 'invModel.txt'):
        
        self.instance.pprint(name)
        
            
    def processResults(self, printOutput = True):
        ''' Prosessing results from pyomo form to pandas data-frames
        for storing and plotting. '''
        
        if printOutput:
            print('Prosessing results from deteministic model...')
        
        model = self.instance
        
        dmr.processRes(self, model)
        
    def saveRes(self, save_dir):    
        ''' Saving prosessed results.  '''
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        pd.DataFrame.from_dict({'sol_time':[self.solution_time]}).to_csv(save_dir + 'sol_time.csv')

        dmr.saveRes(self,save_dir)
        
    def importResults(self, import_dir):
        ''' Importing results from files. '''
        
        dmr.importRes(self, import_dir)

def buildDetModel(mutables = {}):
        ''' Defines abstract investment model, defines sets, parameters, variables,
        constraints and objective function.'''
        
        mutable_dict = {'Power_cost': False,
                        'Energy_cost': False,
                        'CO2_cost': False,
                        'H2_load_scaling': True,
                        'Fuel_price': False,
                        'CCS_cost': False}
        
        for i in mutables.keys():
            mutable_dict[i] = mutables[i]
        
        m = pe.AbstractModel('detModel')
        
        ##Sets##
        m.TIME = pe.Set(ordered = True)
        m.LAST_TIME = pe.Set(ordered = True)
        
        m.NODES = pe.Set()
        m.EL_NODES = pe.Set()
        m.H2_NODES = pe.Set()
        
        m.CURRENT_BRANCHES = pe.Set(dimen = 3)
        m.NEW_BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES_AT_NODE = pe.Set(m.NODES, dimen = 3)
        
        m.CURRENT_LINES = pe.Set(dimen = 3)
        m.NEW_LINES = pe.Set(dimen = 3)
        m.LINES = pe.Set(dimen = 3)
        m.LINES_AT_NODE = pe.Set(m.NODES, dimen = 3)
        
        m.PIPES = pe.Set(dimen = 3)
        m.PIPES_AT_NODE = pe.Set(m.NODES, dimen = 3)
        
        m.PLANT_TYPES = pe.Set()
        m.THERMAL_PLANT_TYPES = pe.Set()
        m.STORAGE_TYPES = pe.Set()
        m.CONV_TYPES = pe.Set()
        
        m.PLANTS = pe.Set()
        m.SOLAR_POWER_PLANTS = pe.Set()
        m.WIND_POWER_PLANTS = pe.Set()
        m.OFFSHORE_WIND_POWER_PLANTS = pe.Set()  
        m.ONSHORE_WIND_POWER_PLANTS = pe.Set()  
        m.POWER_PLANTS = pe.Set()
        m.RENEWABLE_POWER_PLANTS = pe.Set()
        m.THERMAL_POWER_PLANTS = pe.Set()
        m.HYDRO_STORAGE = pe.Set()
        m.SMR_PLANTS = pe.Set()
        m.SMR_CCS_PLANTS = pe.Set()
        m.H2_POWER_PLANTS = pe.Set()
        m.CONV_PLANTS = pe.Set()
        m.GEN_PLANTS = pe.Set()
        
        m.HYDROGEN_STORAGE = pe.Set()        
        m.BATTERY_STORAGE = pe.Set()        
        m.STORAGE = pe.Set()
        m.EL_STORAGE = pe.Set()
        
        m.GEN_AT_NODE = pe.Set(m.NODES)
        m.CONV_AT_NODE = pe.Set(m.NODES)
        m.STORAGE_AT_NODE = pe.Set(m.NODES)
        m.AUX_POWER_AT_NODE = pe.Set(m.NODES)
        
        m.TYPE_TO_PLANTS = pe.Set(m.PLANT_TYPES)
        m.TYPE_TO_THERMAL_PLANTS = pe.Set(m.THERMAL_PLANT_TYPES)
        m.TYPE_TO_STORAGE = pe.Set(m.STORAGE_TYPES)
        
        ##Parameters##
        m.Period_ratio = pe.Param(within = pe.NonNegativeReals)
        
        m.Rationing_cost = pe.Param(within = pe.NonNegativeReals)
        m.CO2_cost = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['CO2_cost'])
        m.Fuel_price = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['Fuel_price'])
        m.CCS_cost = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['CCS_cost'])
        
        
        m.Load = pe.Param(m.TIME, m.NODES, within = pe.NonNegativeReals, default = 0)
        
        m.CO2_coef = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
        m.Power_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, mutable = mutable_dict['Power_cost'])
        m.Energy_cost = pe.Param(m.STORAGE_TYPES, within = pe.NonNegativeReals, mutable = mutable_dict['Energy_cost'])
        m.Fixed_energy_cost = pe.Param(m.STORAGE_TYPES, within = pe.NonNegativeReals)
        m.Fixed_power_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
        m.Var_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
        m.Retirement_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, default = 0)
        m.Retirement_rate = pe.Param(within = pe.NonNegativeReals, default = 0.1)
        m.Fuel_rate = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, default = 0)        
        m.CCS_rate = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, default = 0)

        m.Ramp_rate = pe.Param(m.GEN_PLANTS, within = pe.NonNegativeReals, default = 1)
        m.Max_num_plants = pe.Param(m.GEN_PLANTS, within = pe.NonNegativeReals, default = 100)
        m.Plant_size = pe.Param(m.GEN_PLANTS, within = pe.NonNegativeReals, default = 100)
        m.Min_prod = pe.Param(m.GEN_PLANTS, within = pe.NonNegativeReals, default = 0)
        
        m.Init_power = pe.Param(m.PLANTS, within = pe.NonNegativeReals, default = 0)
        m.Init_energy = pe.Param(m.STORAGE, within = pe.NonNegativeReals, default = 0)
        m.Renewable_pot = pe.Param(m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Energy_max = pe.Param(m.STORAGE, within = pe.NonNegativeReals)
        
        m.Renewable_profile = pe.Param(m.TIME, m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals, default = 0)
        m.Inst_profile = pe.Param(m.TIME, m.ONSHORE_WIND_POWER_PLANTS, within = pe.NonNegativeReals, default = 0) 
        m.Inflow = pe.Param(m.TIME, m.STORAGE, within = pe.NonNegativeReals, default=0) 
        m.Inflow_ureg = pe.Param(m.TIME, m.STORAGE, within = pe.NonNegativeReals, default=0)
        m.Conv_rate = pe.Param(m.CONV_PLANTS, within = pe.NonNegativeReals)
        m.Aux_rate = pe.Param(m.STORAGE, within = pe.NonNegativeReals)
        
        m.Load_scaling = pe.Param(m.NODES, within = pe.NonNegativeReals, default = 1,
                                  mutable = mutable_dict['H2_load_scaling'])
        
        m.Initial_storage = pe.Param(m.STORAGE, within = pe.NonNegativeReals, default = 0)
        
        m.Eff_in = pe.Param(m.STORAGE, within = pe.NonNegativeReals)
        m.Eff_out = pe.Param(m.STORAGE, within = pe.NonNegativeReals)
        
        m.Branch_cap = pe.Param(m.BRANCHES,within = pe.NonNegativeReals)
        m.Branch_cost = pe.Param(m.NEW_BRANCHES,within = pe.NonNegativeReals)
        #m.Susceptance = pe.Param(m.BRANCHES,within = pe.Reals) # Non-Negative?
        m.Branch_dir_at_node = pe.Param(m.NODES,m.BRANCHES, within = pe.Integers)

        #m.Pipe_cap = pe.Param(m.PIPES,within = pe.NonNegativeReals)
        #m.Pipe_cost = pe.Param(m.NEW_PIPES,within = pe.NonNegativeReals)
        #m.Pipe_dir_at_node = pe.Param(m.NODES,m.PIPES, within = pe.Integers)   
             
        # Variables
        m.exp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        m.imp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        
        m.new_power = pe.Var(m.PLANTS, within = pe.NonNegativeReals)
        m.new_energy = pe.Var(m.STORAGE, within = pe.NonNegativeReals)
        
        m.available_plants = pe.Var(m.GEN_PLANTS, within = pe.NonNegativeReals)
        m.retired_cap = pe.Var(m.GEN_PLANTS, within = pe.NonNegativeReals)
        
        m.prod = pe.Var(m.TIME, m.PLANTS, within = pe.NonNegativeReals)
        m.cur = pe.Var(m.TIME, m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals)
        m.state = pe.Var(m.TIME, m.GEN_PLANTS, within = pe.NonNegativeReals)

        m.to_storage = pe.Var(m.TIME, m.STORAGE, within = pe.NonNegativeReals)
        m.from_storage = pe.Var(m.TIME, m.STORAGE, within = pe.NonNegativeReals)
        m.storage = pe.Var(m.TIME, m.STORAGE, within = pe.NonNegativeReals)
        m.spill = pe.Var(m.TIME, m.STORAGE, within = pe.NonNegativeReals)
        
        m.rat = pe.Var(m.TIME, m.NODES, within = pe.NonNegativeReals)
        m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within = pe.Reals)
        #m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
        m.new_branch = pe.Var(m.NEW_BRANCHES, within = pe.Reals, bounds = (0,1))
        
        ## Constraints##
        
        # ALL POWER PLANTS
        
        def numPlants_rule(m,i):
            return m.Plant_size[i]*m.available_plants[i] <= m.Init_power[i] + m.new_power[i] - m.retired_cap[i]
        m.numPlants = pe.Constraint(m.GEN_PLANTS, rule = numPlants_rule)
        
        def genState_rule(m,t,i):
            return m.state[t,i] <= m.available_plants[i]
        m.genState = pe.Constraint(m.TIME, m.GEN_PLANTS, rule = genState_rule)
        
        def maxPlant_rule(m,i):
            return m.available_plants[i] <= m.Max_num_plants[i]
        m.maxPlant = pe.Constraint(m.GEN_PLANTS, rule = maxPlant_rule)
        
        def maxProd_rule(m,t,i):
            return m.prod[t,i]  <= m.Plant_size[i]*m.state[t,i]
        m.maxProd = pe.Constraint(m.TIME,m.GEN_PLANTS, rule = maxProd_rule)
        
        def minProd_rule(m,t,i):
            return m.prod[t,i]  >= m.Min_prod[i]*m.state[t,i]
        m.minProd = pe.Constraint(m.TIME,m.GEN_PLANTS, rule = minProd_rule)
        
        
        def newHydroPower_rule(m,i):
            return m.new_power[i] == 0.0
        m.newHydroPower = pe.Constraint(m.HYDRO_STORAGE, rule = newHydroPower_rule)
        
        def newHydroEnergy_rule(m,i):
            return m.new_energy[i] == 0.0
        m.newHydroEnergy = pe.Constraint(m.HYDRO_STORAGE, rule = newHydroEnergy_rule)
        
        def minHydroProd_rule(m,t,i):
            return m.from_storage[t,i] + m.spill[t,i] >= m.Inflow_ureg[t,i]
        m.minHydroProd = pe.Constraint(m.TIME, m.HYDRO_STORAGE, rule = minHydroProd_rule)
        
#        def startUp_rule(m,t,i):
#            return m.state[t,i] == m.state[t-1,i] + m.start_up[t,i] - m.shut_down[t,i] 
#        m.startUp = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = startUp_rule)
        
        def rampUpLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t,i] - m.prod[t-1,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampUpLimit = pe.Constraint(m.TIME,m.GEN_PLANTS, rule = rampUpLimit_rule)
        
        def rampDownLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t-1,i] - m.prod[t,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampDownLimit = pe.Constraint(m.TIME,m.GEN_PLANTS, rule = rampDownLimit_rule)
            
#        def uptime_rule(m,t,i):
#            
#            
#            
#        def downtime_rule(m,t,i):
        
        def maxRenewable_rule(m,i):
            return m.new_power[i]  <= m.Renewable_pot[i]
        m.maxSolarCap = pe.Constraint(m.RENEWABLE_POWER_PLANTS, rule = maxRenewable_rule)
                 
        # Renewable power balance
        def renewableBalance_rule(m,t,i):
            if i in m.ONSHORE_WIND_POWER_PLANTS:
                return m.prod[t,i] + m.cur[t,i] == m.Inst_profile[t,i]*m.Init_power[i] + m.Renewable_profile[t,i]*m.new_power[i]                    
            else:
                return m.prod[t,i] + m.cur[t,i] == m.Renewable_profile[t,i]*(m.Init_power[i] + m.new_power[i])
        m.renewableBalance = pe.Constraint(m.TIME,m.RENEWABLE_POWER_PLANTS,
                                      rule = renewableBalance_rule)   
                
#       Storage plants
        # Storage loss occurs at the storage side and is represented by a
        # efficiency, eta = (1-loss) where eta^in = eat^out = sqrt(eta)
        def storageBalance_rule(m,t,i):
            if t == 0:
                return m.storage[t,i] == m.Initial_storage[i]*(m.Init_energy[i] \
                                       + m.new_energy[i]) + m.Eff_in[i]*m.to_storage[t,i] \
                                       - m.from_storage[t,i]/m.Eff_out[i] + m.Inflow[t,i] \
                                       + m.Inflow_ureg[t,i] - m.spill[t,i]
            else:
                return m.storage[t,i] == m.storage[t-1,i] \
                             + m.Eff_in[i]*m.to_storage[t,i] - m.from_storage[t,i]/m.Eff_out[i] \
                             + m.Inflow[t,i] + m.Inflow_ureg[t,i] - m.spill[t,i]
        m.storageBalance = pe.Constraint(m.TIME, m.STORAGE, rule = storageBalance_rule)
        
        def endStorage_rule(m,t,i):
            return m.storage[t,i] == m.Initial_storage[i]*(m.Init_energy[i] + m.new_energy[i])
        m.endStorage = pe.Constraint(m.LAST_TIME, m.STORAGE, rule = endStorage_rule)
        
        def storageEnergyCap_rule(m, t, i):
            return m.storage[t,i] <= m.Init_energy[i] + m.new_energy[i]
        m.storageEnergyCap = pe.Constraint(m.TIME, m.STORAGE, rule = storageEnergyCap_rule)
        
        def storageInPowerCap_rule(m, t, i):
            return m.to_storage[t,i] <= m.Init_power[i] + m.new_power[i]
        m.storageInPowerCap = pe.Constraint(m.TIME, m.STORAGE, rule = storageInPowerCap_rule)

        def storageOutPowerCap_rule(m, t, i):
            return m.from_storage[t,i] <= m.Init_power[i] + m.new_power[i]
        m.storageOutPowerCap = pe.Constraint(m.TIME, m.STORAGE, rule = storageOutPowerCap_rule)
            
        # def maxStorageCap_rule(m,i):
        #     return m.new_energy[i]  <= m.Energy_max[i]
        # m.maxStorageCap = pe.Constraint(m.STORAGE, rule = maxStorageCap_rule)
                        
        # Energy balance
        def energyBalance_rule(m,t,i):
            return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                    + sum(m.from_storage[t,j] - m.to_storage[t,j] for j in m.STORAGE_AT_NODE[i]) \
                    + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] == m.Load[t,i]*m.Load_scaling[i] \
                    + sum(m.Conv_rate[j]*m.prod[t,j] for j in m.CONV_AT_NODE[i]) \
                    + sum(m.Aux_rate[j]*m.to_storage[t,j] for j in m.AUX_POWER_AT_NODE[i])
        m.energyBalance = pe.Constraint(m.TIME, m.NODES, rule = energyBalance_rule)
             
        # DC power flow
#        def referenceNode_rule(m,t):
#            return m.voltage_angle[t,m.NODES[1]] == 0.0
#        m.ref_node = pe.Constraint(m.TIME, rule = referenceNode_rule)
        
#        def branchFlow_rule(m,t,n,i,j):
#            return m.branch_flow[t,n,i,j] == m.Susceptance[n,i,j]*(m.voltage_angle[t,i]-m.voltage_angle[t,j])
#        m.branchFlow = pe.Constraint(m.TIME, m.BRANCHES, rule = branchFlow_rule)
        
        def branchFlowLimit_rule(m,t,n,i,j):
            if not np.isinf(m.Branch_cap[n,i,j]):
                return (-m.Branch_cap[n,i,j], m.branch_flow[t,n,i,j], m.Branch_cap[n,i,j])
            else:
                return (-10000, m.branch_flow[t,n,i,j], 10000)
        m.branchFlowLimit = pe.Constraint(m.TIME, m.CURRENT_LINES, rule = branchFlowLimit_rule )
        
        def newBranchFlowUpperLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Branch_cap[n, i,j])):
                return m.branch_flow[t,n,i,j] <= m.new_branch[n,i,j]*m.Branch_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] <= 10000
        m.newBranchFlowUpperLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES, rule = newBranchFlowUpperLimit_rule )
        
        def newBranchFlowLowerLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Branch_cap[n,i,j])):
                return m.branch_flow[t,n,i,j] >= -m.new_branch[n,i,j]*m.Branch_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] >= -10000
        m.newBranchFlowLowerLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES, rule = newBranchFlowLowerLimit_rule )
        
        def nodalBalance_rule(m,t,i):
            return m.imp[t,i] - m.exp[t,i] == sum(m.Branch_dir_at_node[i,j]*m.branch_flow[t,j] for j in m.BRANCHES_AT_NODE[i])
        m.nodalBalance = pe.Constraint(m.TIME, m.NODES, rule = nodalBalance_rule)
        
        def obj_rule(m):
            return  m.Period_ratio*(sum(sum(m.Power_cost[j]*m.new_power[i]  for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)\
                                    + sum(sum((m.Energy_cost[j] + m.Fixed_energy_cost[j])*m.new_energy[i] + m.Fixed_power_cost[j]*m.new_power[i] for i in m.TYPE_TO_STORAGE[j]) for j in m.STORAGE_TYPES)\
                    + sum(sum(m.Fixed_power_cost[j]*m.Plant_size[i]*m.available_plants[i] + m.Retirement_rate*m.Power_cost[j]*m.retired_cap[i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)\
                    + sum(m.Branch_cost[i]*m.new_branch[i] for i in m.NEW_BRANCHES)) \
                    + sum(sum(sum((m.Fuel_price*m.Fuel_rate[j] + m.Var_cost[j] + m.CO2_cost*m.CO2_coef[j] + m.CCS_cost*m.CCS_rate[j])*m.prod[t,i] for i in m.TYPE_TO_PLANTS[j])for j in m.PLANT_TYPES)
                          + sum(m.Rationing_cost*m.rat[t,i] for i in m.NODES) for t in m.TIME)
        m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
        
        #####
        
#        m.cut = pe.ConstraintList()
##        def cut_rule(m,k):
##            return m.alpha >= m.W[k] +sum(m.Const[i,k] for i in m.PLANTS) \
##                            - sum(m.Power_grad[i,k]*m.new_power[i] for i in m.PLANTS) \
##                            - sum(m.Energy_grad[i,k]*m.new_power[i] for i in m.STORAGE)
#        
#        def obj_inv_rule(m):
#            return  m.Period_ratio*(sum(sum(m.Power_cost[j]*m.new_power[i]  for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)\
#                                    + sum(sum((m.Energy_cost[j] + m.Fixed_energy_cost[j])*m.new_energy[i] + m.Fixed_power_cost[j]*m.new_power[i] for i in m.TYPE_TO_STORAGE[j]) for j in m.STORAGE_TYPES)\
#                    + sum(sum(m.Fixed_power_cost[j]*m.Plant_size[i]*m.available_plants[i] + m.Retirement_cost[j]*m.retired_cap[i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)\
#                    + sum(m.Branch_cost[i]*m.Branch_cap[i]*m.new_branch[i] for i in m.NEW_BRANCHES))\
#                    + m.alpha
#        m.obj_inv = pe.Objective(rule = obj_inv_rule, sense = pe.minimize)
        
        
        
        return m



def detData(obj):
    ''' Data input for the investment model according to the structure of the
    abstract model.'''
    di = {}
    ##Set##
    di['TIME'] = {None: list(obj.timerange)}
    di['LAST_TIME'] = {None: [list(obj.timerange)[-1]]}
    # Set node structure 
    node_data = copy.copy(obj.data.bus)
    el_pf = 'EN'
    h2_pf = 'HN'
    
    di['EL_NODES'] = {None: [el_pf + '%.2d' % i for i in node_data.Bus.tolist()]}
    di['H2_NODES'] = {None: [h2_pf + '%.2d' % i for i in node_data.Bus.tolist()]}
    di['NODES'] = {None: di['EL_NODES'][None] + di['H2_NODES'][None]}
    # Set line structure
    line_data = copy.copy(obj.data.line)
    branch_indx = []
    new_branch_indx = []
    pipe_indx = []
    for i in line_data.index:
        from_bus = '%.2d' % line_data.From[i]
        to_bus = '%.2d' % line_data.To[i]
        if line_data.Type[i] == 'Existing':
            n_typ = el_pf
            branch_indx.append((i,n_typ + from_bus, n_typ + to_bus))
        elif line_data.Type[i] == 'New':
            n_typ = el_pf
            new_branch_indx.append((i,n_typ + from_bus, n_typ + to_bus))
        elif line_data.Type[i] == 'H2':
            n_typ = h2_pf
            pipe_indx.append((i, n_typ + from_bus, n_typ + to_bus))
        
    di['CURRENT_LINES'] = {None: branch_indx }
    di['NEW_LINES'] = {None: new_branch_indx }
    di['LINES'] =  {None: di['CURRENT_LINES'][None] + di['NEW_LINES'][None]}
    
    di['PIPES'] =  {None: pipe_indx }
    
    di['BRANCHES'] =  {None: di['LINES'][None] + di['PIPES'][None]}
    di['NEW_BRANCHES'] ={None: di['NEW_LINES'][None] + di['PIPES'][None]}

    def getBranchesAtNode(set_type):
        out = {}
        for node in di['NODES'][None]:
            for n,i,j in di[set_type][None]:
                if i == node or j == node:
                    if node not in out.keys():
                        out[node] = []
                    out[node].append((n,i,j))
        return out
    di['BRANCHES_AT_NODE'] = getBranchesAtNode('BRANCHES')

    # Define unit sets
    installed = copy.copy(obj.data.installed)
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('Type', inplace = True)
    plant_char.fillna(0, inplace = True)
    
    storage_char = copy.copy(obj.data.storage_char)
    storage_char.set_index('Type', inplace = True)
    storage_char.fillna(0, inplace = True)
    
    di['PLANT_TYPES'] = {None: plant_char.index.to_list() + storage_char.index.to_list()}
    power_plants = plant_char.index[obj.data.plant_char.Class.isin(['RE','TH','H2TH'])]
    di['POWER_PLANT_TYPES'] = {None: power_plants.to_list()}
    thermal_plants = plant_char.index[obj.data.plant_char.Class.isin(['TH','H2TH'])]
    di['THERMAL_PLANT_TYPES'] = {None: thermal_plants.to_list()}
    hydrogen_plants = plant_char.index[obj.data.plant_char.Class.isin(['H2'])]
    di['HYDROGEN_PLANT_TYPES'] = {None: hydrogen_plants.to_list()}
    di['STORAGE_TYPES'] = {None: storage_char.index.to_list()}
    h2_conv_plants = plant_char.index[obj.data.plant_char.Class.isin(['H2TH'])]
    di['CONV_TYPES'] = {None:['PEMEL'] + h2_conv_plants.to_list()}
    di['GAS_PLANT_TYPES'] = {None: ['CC Gas','CT Gas','CCS Gas']}
    di['SMR_PLANT_TYPES'] = {None: ['SMR','SMR CCS']}
    
    
    obj.type2prefix = {'Biomass' : 'B', 'CC Gas' : 'CCG', 'CT Gas' : 'CTG',
                             'CCS Gas' : 'CCSG', 'CT H2' : 'CTH', 'CC H2' : 'CCH',
                            'Coal' : 'C', 'CCS Coal' : 'CCSC', 'Nuclear' : 'N',
                            'Solar' : 'S', 'Offshore Wind' : 'OW',
                             'Onshore Wind' : 'SW', 'Wind' : 'W',
                              'Hydrogen': 'HS', 'Battery': 'BS',
                              'PEMEL': 'PEMEL', 'PEMFC':'PEMFC',
                              'SMR':'SMR','SMR CCS': 'SMRCCS',
                            'Hydro Power': 'HP'} # 'ICE Gas' : 'ICEG','SOFC':'SOFC', 'ICE H2' : 'ICEH','SOFC':'SOFC',
    
    di['type2prefix'] = obj.type2prefix

    # Create individual plant set and set for plant-type to individual plants
    di['TYPE_TO_PLANTS'] = {}
    di['TYPE_TO_THERMAL_PLANTS'] = {}
    di['TYPE_TO_CONV_PLANTS'] = {}
    di['PLANTS'] = {None: []}
    for k in di['PLANT_TYPES'][None]:
        if k in di['POWER_PLANT_TYPES'][None]:
            class_type = '_POWER_PLANTS'
        elif k in di['HYDROGEN_PLANT_TYPES'][None]:
            class_type = '_PLANTS'
        elif k in di['STORAGE_TYPES'][None]:
            class_type = '_STORAGE'
            
        set_name = k.replace(' ','_').upper() + class_type
        di[set_name] = {None: [obj.type2prefix[k] + '%.2d' % i for i in node_data.Bus.tolist()]}
        di['PLANTS'][None] += [obj.type2prefix[k] + '%.2d' % i for i in node_data.Bus.tolist()]
        
        if class_type == '_PLANTS':
            class_type = '_H2_PLANTS'
        if not 'TYPE_TO' + class_type in di.keys():
            di['TYPE_TO' + class_type] = {}
        di['TYPE_TO' + class_type][k] = di[set_name][None]
        di['TYPE_TO_PLANTS'][k] = di[set_name][None]
        if k in di['THERMAL_PLANT_TYPES'][None]:
            di['TYPE_TO_THERMAL_PLANTS'][k] = di[set_name][None]
        if k in  di['CONV_TYPES'][None]:
            di['TYPE_TO_CONV_PLANTS'][k] = di[set_name][None]
            
    di['TYPE_TO_GEN_PLANTS'] = {**di['TYPE_TO_THERMAL_PLANTS'], **di['TYPE_TO_H2_PLANTS']}
            

    
    #-- COLLECTIONS (higher level sets/categories)--
    # All wind power plants
    di['WIND_POWER_PLANTS'] = {None: di['OFFSHORE_WIND_POWER_PLANTS'][None]
                                     + di['ONSHORE_WIND_POWER_PLANTS'][None]}
    di['RENEWABLE_POWER_PLANTS'] = {None: di['WIND_POWER_PLANTS'][None] \
      + di['SOLAR_POWER_PLANTS'][None]}
    # Storage of electricity only
    di['EL_STORAGE'] = {None: di['BATTERY_STORAGE'][None]}# + di['HYDRO_STORAGE'][None]}
    # All storage
    di['STORAGE'] = {None: di['HYDROGEN_STORAGE'][None] + di['EL_STORAGE'][None]}
    di['GAS_POWER_PLANTS'] = {None: di['CC_GAS_POWER_PLANTS'][None] \
      + di['CT_GAS_POWER_PLANTS'][None] + di['CCS_GAS_POWER_PLANTS'][None]}
    # Plants generating el from h2
    di['H2_POWER_PLANTS'] = {None: di['CC_H2_POWER_PLANTS'][None]\
      + di['CT_H2_POWER_PLANTS'][None]  \
      + di['PEMFC_POWER_PLANTS'][None]}#+ di['ICE_H2_POWER_PLANTS'][None] + di['SOFC_POWER_PLANTS'][None] }
    di['THERMAL_POWER_PLANTS'] = {None: di['BIOMASS_POWER_PLANTS'][None] \
      + di['COAL_POWER_PLANTS'][None] + di['GAS_POWER_PLANTS'][None] \
      + di['NUCLEAR_POWER_PLANTS'][None] + di['CCS_COAL_POWER_PLANTS'][None] \
      + di['H2_POWER_PLANTS'][None] }#+ di['ICE_GAS_POWER_PLANTS'][None] }
    # Plants generating h2
    di['H2_PLANTS'] = {None: di['PEMEL_PLANTS'][None] \
      + di['SMR_PLANTS'][None] + di['SMR_CCS_PLANTS'][None]}
    # All plants generating el
    di['POWER_PLANTS'] = {None: di['RENEWABLE_POWER_PLANTS'][None] \
      + di['THERMAL_POWER_PLANTS'][None]}
    di['GEN_PLANTS'] = {None: di['H2_PLANTS'][None] \
      + di['THERMAL_POWER_PLANTS'][None]}
    di['CONV_PLANTS'] = {None: di['PEMEL_PLANTS'][None] + di['H2_POWER_PLANTS'][None]}
    
    load_series = copy.copy(obj.data.load_series)
    load_series.columns = [el_pf + '%.2d' % int(i) for i in load_series.columns]
    load_series = load_series[load_series.index.isin(obj.time)]
    load_series.index = list(obj.timerange)
    
    # -- Position of units in the networks --
    # Platns generating el at el nodes      
    di['GEN_AT_NODE'] = {i:[j for j in di['POWER_PLANTS'][None]
                    if j[-2:] == i[-2:]] for i in di['EL_NODES'][None]}
    
    # Plants generating h2 at h2 node
    di['GEN_AT_NODE'].update({i :[j for j in (di['H2_PLANTS'][None])
                    if j[-2:] == i[-2:]] for i in di['H2_NODES'][None]})
    
    # Plants converting h2 into el at h2 nodes
    di['CONV_AT_NODE'] = {i:[j for j in (di['H2_POWER_PLANTS'][None])
                    if j[-2:] == i[-2:]] for i in di['H2_NODES'][None]}
    # Plants converting el into h2 at el nodes
    di['CONV_AT_NODE'].update({i:[j for j in (di['PEMEL_PLANTS'][None])
                    if j[-2:] == i[-2:]] for i in di['EL_NODES'][None]})
    
    di['BATTERY_STORAGE_AT_NODE'] = {i :[j for j in (di['BATTERY_STORAGE'][None])
                    if j[-2:] == i[-2:]] for i in di['NODES'][None]}
    
#    di['HYDRO_POWER_AT_NODE'] = {i :[j for j in (di['HYDRO_STORAGE'][None])
#                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
    # El storage at el nodes
    di['STORAGE_AT_NODE'] = {i :[j for j in (di['EL_STORAGE'][None])
                    if j[-2:] == i[-2:]] for i in di['EL_NODES'][None]}
    # H2 storage at h2 nodes
    di['STORAGE_AT_NODE'].update({i :[j for j in (di['HYDROGEN_STORAGE'][None])
                    if j[-2:] == i[-2:]] for i in di['H2_NODES'][None]})
    
    di['AUX_POWER_AT_NODE']= {i :[j for j in (di['HYDROGEN_STORAGE'][None])
                    if j[-2:] == i[-2:]] for i in di['EL_NODES'][None]}
    
    
    ##Parameters##
    di['NTime'] = {None: len(obj.timerange)}
    di['Period_ratio'] = {None: len(obj.timerange)/8760}
    
    param = copy.copy(obj.data.parameters)
    
    di['Load'] = load_series.stack().to_dict()
    
    h2_load = pd.DataFrame(index = obj.data.load_series.index,
                           columns = di['H2_NODES'][None])
    for i in di['H2_NODES'][None]:
        indx = obj.data.hydrogen_load.Bus == int(i[-2:])
        value = obj.data.hydrogen_load.loc[indx,'high'].values/24 #kg/h
        #value = [0.0]
        if len(value) > 0:
            h2_load.loc[:,i] = value[0]    
    h2_load.fillna(0, inplace = True)
    h2_load = h2_load[h2_load.index.isin(obj.time)]
    h2_load.index = np.arange(len(h2_load.index))
    di['Load'].update(h2_load.stack().to_dict())

    installed.set_index('Bus', inplace = True)
    init_cap = installed
    init_cap.fillna(0, inplace = True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (obj.type2prefix[j],i) : init_cap[i,j] for i,j in init_cap.keys()}
    di['Init_power'] = init_cap_dict
    
    init_energy = copy.copy(obj.data.installed_energy)
    init_energy.set_index('Bus', inplace = True)
    init_energy.fillna(0, inplace = True)
    init_energy = init_energy.stack().to_dict()
    init_energy_dict = {'%s%.2d' % (obj.type2prefix[j],i) : init_energy[i,j] for i,j in init_energy.keys()}

    di['Init_energy'] = init_energy_dict
          
    ramp_rate = copy.copy(plant_char['Ramp (%/h)'])
    rate = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys(): 
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            rate[p] = ramp_rate.loc[t]
    di['Ramp_rate'] = rate # %/h
    
    min_limit = copy.copy(plant_char['Min Gen (pu)'])
    m_lim = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys(): 
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            m_lim[p] = min_limit.loc[t]
    di['Min_prod'] = m_lim # MW
    
    max_num_plants = copy.copy(obj.data.max_num_plants)
    max_num_plants.set_index('Type', inplace = True)
    max_plants = {}
    for t in max_num_plants.index: 
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            max_plants[p] = max_num_plants.plants.loc[t]
    di['Max_num_plants'] = max_plants # MW
    
    plant_size = plant_char['Typical Plant Size (pu)']
    p_size = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys(): 
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            p_size[p] = plant_size.loc[t]
    di['Plant_size'] = p_size # MW
    
    Power_cost = plant_char['Inv ($/pu-year)']
    di['Power_cost'] = Power_cost.to_dict()
    di['Power_cost'].update(storage_char['Inv power ($/pu-yr)'].to_dict())
    
    di['Energy_cost'] = storage_char['Inv energy ($/eu-yr)'].to_dict()
    
    Fixed_power_cost = plant_char['Fix ($/pu-year)']
    di['Fixed_power_cost'] = Fixed_power_cost.to_dict()
    di['Fixed_power_cost'].update(storage_char['Fix power ($/pu-yr)'].to_dict())
    
    di['Fixed_energy_cost'] = storage_char['Fix energy ($/eu-yr)'].to_dict()
    
    var_cost = plant_char['Var ($/eu)']
    di['Var_cost'] = var_cost.to_dict()
    di['Var_cost'].update(storage_char['Var pow ($/eu)'].to_dict())
    
    fuel_rate = plant_char['Fuel (in eu/out eu)']
    #fuel_rate = fuel_rate[fuel_rate.index.isin(di['CONV_TYPES'][None])]
    fuel_rate_dict = fuel_rate.to_dict()
    di['Conv_rate'] = {}
    for k,v in di['TYPE_TO_CONV_PLANTS'].items():
        di['Conv_rate'].update({i: fuel_rate_dict[k] for i in v})
    
    di['Fuel_rate'] = {}
    fuel_types = di['GAS_PLANT_TYPES'][None] + di['SMR_PLANT_TYPES'][None]
    for i in fuel_types:
        di['Fuel_rate'].update({i: fuel_rate_dict[i]})
    
    aux_rate = storage_char['Aux power (MWh/eu)']
    aux_rate_dict = aux_rate.to_dict()
    di['Aux_rate'] = {}
    for k,v in di['TYPE_TO_STORAGE'].items():
        di['Aux_rate'].update({i: aux_rate_dict[k] for i in v})
    
    retirement_cost = copy.copy(obj.data.retirement_cost)
    retirement_cost.index = obj.data.retirement_cost.Type
    di['Retirement_cost'] = retirement_cost.Cost.to_dict()
    
    CO2_coef = copy.copy(plant_char['Emission (kg/eu)'])
    di['CO2_coef'] = CO2_coef.to_dict()
    di['CO2_coef'].update(storage_char['Emission (kg/eu)'].to_dict())
    
    CCS_rate = copy.copy(plant_char['CCS rate (kg/eu)'])
    CCS_rate = CCS_rate[CCS_rate > 0]
    di['CCS_rate'] = CCS_rate.to_dict()
    
    solar_series = copy.copy(obj.data.solar_series)
    solar_series = solar_series[solar_series.index.isin(obj.time)]
    solar_series.index = pd.Index(np.arange(len(solar_series.index)))
    solar_series.rename(columns = {i : obj.type2prefix['Solar'] + '%.2d' % int(i) for i in solar_series.columns.tolist()},
                                  level = 0, inplace = True)
    solar_series[solar_series < 0] = 0.0
    di['Renewable_profile'] = solar_series.round(4).stack(level = 0).to_dict()
	    
    wind_series = copy.copy(obj.data.wind_series)
    wind_series = wind_series[wind_series.index.isin(obj.time)]
    wind_series.index = np.arange(len(wind_series.index))
    wind_series.rename(columns = {i : obj.type2prefix['Onshore Wind'] + '%.2d' % int(i) for i in wind_series.columns.tolist()},
                                  inplace = True)
    wind_series[wind_series < 0] = 0.0
    di['Renewable_profile'].update(wind_series.round(4).fillna(0).unstack().swaplevel().to_dict())
    
    offshore_wind_series = copy.copy(obj.data.offshore_wind_series)
    offshore_wind_series = offshore_wind_series[offshore_wind_series.index.isin(obj.time)]
    offshore_wind_series.index = np.arange(len(offshore_wind_series.index))
    offshore_wind_series.rename(columns = {i : obj.type2prefix['Offshore Wind'] + '%.2d' % int(i) for i in offshore_wind_series.columns.tolist()},
                                  inplace = True)
    offshore_wind_series[offshore_wind_series < 0] = 0.0
    di['Renewable_profile'].update(offshore_wind_series.round(4).fillna(0).unstack().swaplevel().to_dict())
    
    inst_wind_series = copy.copy(obj.data.inst_wind_series)
    inst_wind_series = inst_wind_series[inst_wind_series.index.isin(obj.time)]
    inst_wind_series.index = np.arange(len(inst_wind_series.index))
    inst_wind_series.rename(columns = {i : obj.type2prefix['Onshore Wind'] + '%.2d' % int(i) for i in inst_wind_series.columns.tolist()},
                                  inplace = True)
    inst_wind_series[inst_wind_series < 0] = 0.0
    di['Inst_profile']= inst_wind_series.round(4).fillna(0).unstack().swaplevel().to_dict()


    if hasattr(obj.data, 'inflow_series'):
        inflow_series = copy.copy(obj.data.inflow_series)
        inflow_series.index = np.arange(len(inflow_series.index))
        inflow_series = inflow_series[inflow_series.index.isin(di['TIME'][None])]
        inflow_series.rename(columns = {i : obj.type2prefix['Hydro Power'] + '%.2d' % int(i) for i in inflow_series.columns.tolist()},
                                      inplace = True)
        inflow_series[inflow_series < 0] = 0.0
        di['Inflow'] = inflow_series.round(4).fillna(0).unstack().swaplevel().to_dict()
        
        inflow_ureg_series = copy.copy(obj.data.inflow_ureg_series)
        inflow_ureg_series.index = np.arange(len(inflow_ureg_series.index))
        inflow_ureg_series = inflow_ureg_series[inflow_ureg_series.index.isin(di['TIME'][None])]
        inflow_ureg_series.rename(columns = {i : obj.type2prefix['Hydro Power'] + '%.2d' % int(i) for i in inflow_ureg_series.columns.tolist()},
                                      inplace = True)
        inflow_ureg_series[inflow_ureg_series < 0] = 0.0
        di['Inflow_ureg'] = inflow_ureg_series.round(4).fillna(0).unstack().swaplevel().to_dict()
    else:
        di['Inflow'] = {}
        di['Inflow_ureg'] = {}
    
         
    renewable_pot = copy.copy(obj.data.renewable_pot)
    renewable_pot.set_index('Bus', inplace = True)
    renewable_pot.fillna(0, inplace = True)
    renewable_pot = renewable_pot.stack().to_dict()
    renewable_pot_dict = {'%s%.2d' % (obj.type2prefix[j],i) : renewable_pot[i,j] for i,j in renewable_pot.keys()}
    di['Renewable_pot'] = renewable_pot_dict

    eff_in = {}
    eff_out = {}
    en_max = {}
    for k,v in di['TYPE_TO_STORAGE'].items():
        for i in v:
            eff_in[i] = storage_char.loc[k,'In (%)']
            eff_out[i] = storage_char.loc[k,'Out (%)']
            en_max[i] = storage_char.loc[k,'New Energy Max (eu)']
    
    di['Eff_in'] = eff_in
    di['Eff_out'] = eff_out
    di['Energy_max'] = en_max
    di['Initial_storage'] = {i: 0.0 for i in di['STORAGE'][None]}
    
    di['Load_scaling'] = {i: param['H2_scaling'].values[0] for i in di['H2_NODES'][None]} 
    
    new_indx = []
    for i in line_data.index:
        ltp = line_data.iloc[i].Type
        fn = '%.2d' % line_data.iloc[i].From
        tn = '%.2d' % line_data.iloc[i].To
        if ltp == 'H2':
            ntp = h2_pf
        else:
            ntp = el_pf
        new_indx.append((i,ntp + fn,ntp + tn))
        
    line_data.index = new_indx
    di['Branch_cost'] = line_data[line_data.Type.isin(['New','H2'])].Cost.to_dict()
    di['Branch_cap'] = line_data.Cap.to_dict()
    #di['Susceptance'] = line_data.B.to_dict()

    di['Rationing_cost'] = {None: param.at[0,'rat_cost']}
    di['CO2_cost'] = {None: param.at[0,'CO2_cost']}
    di['Fuel_price'] = {None: param.at[0,'NG price ($/mmBtu)']}
    di['CCS_cost'] = {None: float(param.at[0,'CCS cost ($/kg)'])}

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
