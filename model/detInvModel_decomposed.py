# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:07:41 2017

@author: espenfb
"""

import pyomo.environ as pe
import os
import detModelRes_decomposed as dmr
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
    ''' Deterministic model for regional power system with hydogen loads,
    wind power and hydro power. '''
    
    def __init__(self, time_data, dirs, mutables = {}):
         
        
        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])
        
        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])
            
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
            
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
        self.bound = pd.DataFrame()
        
    def buildInvModel(self):
        
        # Connect to solver
        self.opt = pe.SolverFactory('gurobi', solver_io='python')
#        self.opt_opr = pe.SolverFactory('gurobi_persistent', solver_io='python')
#        self.opt_inv = pe.SolverFactory('gurobi_persistent', solver_io='python')
        
        print('Building investment model...')
        self.invModel = buildInvModel(mutables = self.mutables)
    
        # Create concrete instance
        self.invDataInstance = invData(self)
        #print('Creating LP problem instance...')
        self.invModelInstance = self.invModel.create_instance(
                                data= self.invDataInstance,
                                name="Investment model",
                                namespace='invData')
        
        self.cut_w = pd.DataFrame()
        self.cut_const = pd.DataFrame()
        self.cut_grad = pd.DataFrame()
        
    
    def buildOprModel(self):
        
        print('Building operation model...')
        self.oprModel = buildOprModel(mutables = self.mutables)
    
        # Create concrete instance
        self.oprDataInstance = oprData(self)
        #print('Creating LP problem instance...')
        self.oprModelInstance = self.oprModel.create_instance(
                                data= self.oprDataInstance,
                                name="Operation model",
                                namespace='oprData')
        
        # Enable access to duals
        self.oprModelInstance.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        
#        self.opt_opr.set_instance(self.oprModelInstance)
        
    def solveInvModel(self, printOutput = True):
    
        if printOutput:
                print('Solving investment model...')
            
        # Solve model
        start_time = time.time()
        self.pyomo_res = self.opt.solve(self.invModelInstance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True)#,
                      #  options={"NodeMethod": 2,
                      #          "MIPGap": 0.001,
                      #           "MIPFocus": 3})
#        self.pyomo_res = self.opt_inv.solve(tee=printOutput, #stream the solver output
#                        keepfiles=False, #print the LP file for examination
#                        symbolic_solver_labels=True,
#                        save_results=False)#,
    
        #self.detModelInstance.write('model.mps',
        #                        io_options={'symbolic_solver_labels':True})

        self.inv_sol_time = time.time()-start_time
        
    def solveOprModel(self, printOutput = True):
    
        if printOutput:
                print('Solving operation model...')
            
        # Solve model
        start_time = time.time()
        self.pyomo_res = self.opt.solve(self.oprModelInstance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True)#,
                        #options={"Method": 2,
                        #         "Crossover": 0})
#        self.pyomo_res = self.opt_opr.solve(tee=printOutput, #stream the solver output
#                        keepfiles=False, #print the LP file for examination
#                        symbolic_solver_labels=True,
#                        save_results=False)
    
        #self.detModelInstance.write('model.mps',
        #                        io_options={'symbolic_solver_labels':True})

        self.opr_sol_time = time.time()-start_time
        
    def run(self, maxItr = 50, printOutput = True):
        
        self.maxItr = maxItr
        
        self.buildInvModel()
        self.buildOprModel()
#        self.oprModelInstance.branchFlow.activate()
        
        for i in range(self.maxItr):
            
            self.itr = i
            
            print('Itr %d:' % self.itr)
            
            self.solveInvModel()
            print('...solution time: %.2f' % self.inv_sol_time)
            
            print('...updating operational model')
            self.updateOprModel()
            
            self.solveOprModel()
            print('...solution time: %.2f' % self.opr_sol_time)
            
            print('...calculating bounds')
            self.calcBounds()
            
            print('...processing results')
            self.processResults()
            
            print('...saving results')
            self.saveRes(self.res_dir)
            
            print('...adding cuts')
            self.addCut()
            
            print('Plant investments:')
            p_inv = self.inv_res['plant'].sort_index()
            print(p_inv[p_inv.new_cap >0])
            
            print('Line investments:')
            l_inv = self.inv_res['line'].sort_index()
            print(l_inv[l_inv.new_branch_cap >0])
        
        
        
    def printModel(self, name = 'Model.txt'):
        
        self.oprModelInstance.pprint('opr' + name)
        self.invModelInstance.pprint('inv' + name)
        
            
    def processResults(self, printOutput = True):
        ''' Prosessing results from pyomo form to pandas data-frames
        for storing and plotting. '''
        
        if printOutput:
            print('Prosessing results from model...')
        
        oprModel = self.oprModelInstance
        invModel = self.invModelInstance
        
        dmr.processOprRes(self, oprModel)
        dmr.processInvRes(self, invModel)
        
    def saveRes(self, save_dir):    
        ''' Saving prosessed results.  '''
        
        #self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        pd.DataFrame.from_dict({'inv_sol_time':[self.inv_sol_time]}).to_csv(save_dir + 'inv_sol_time.csv')
        pd.DataFrame.from_dict({'opr_sol_time':[self.opr_sol_time]}).to_csv(save_dir + 'opr_sol_time.csv')

        dmr.saveInvRes(self, save_dir + 'Investment\\' + 'Itr%2.d\\' % self.itr)
        dmr.saveOprRes(self, save_dir + 'Operation\\' + 'Itr%2.d\\' % self.itr)
        
    def importResults(self, import_dir):
        ''' Importing results from files. '''
        
        dmr.importDetRes(self, import_dir)
        
    def calcBounds(self):
        
        oprModel = self.oprModelInstance
        invModel = self.invModelInstance
        
        inv_obj = invModel.obj.expr()
        opr_obj = oprModel.obj.expr()
        alpha = invModel.alpha.value
        
        self.bound.loc[self.itr,'Lower'] = inv_obj
        self.bound.loc[self.itr,'Upper'] = inv_obj - alpha + opr_obj
        
        self.bound.to_csv(self.res_dir + 'bounds.csv')
        
    def addCut(self):
        
        model = self.oprModelInstance
        
        self.cut_w.loc[self.itr,'w']  = model.obj.expr()
        self.cut_w = self.cut_w.round(6)
        
        time = model.TIME
        
        prod_coeff = pd.DataFrame()
        for i in model.RENEWABLE_POWER_PLANTS:
            if i in model.SOLAR_POWER_PLANTS:
                constr = model.solarBalance
                prod_series = model.Solar_profile
            elif i in model.WIND_POWER_PLANTS:
                constr = model.windBalance
                prod_series = model.Wind_profile_pot
            for t in time:
                dual = model.dual[constr[t,i]]
                prod_coeff.loc[t,i] = dual*prod_series[t,i]
        for i in model.THERMAL_POWER_PLANTS:
            constr = model.genState
            for t in time:
                dual = model.dual[constr[t,i]]
                prod_coeff.loc[t,i] = dual
        for i in model.STORAGE_PLANTS:
            e_id = model.STORAGE_ENERGY_AT_PLANT[i].value_list[0]
            p_id = model.STORAGE_POWER_AT_PLANT[i].value_list[0]
            constr1e = model.storageEnergyCap
            constr1p = model.storageInPowerCap
            constr2p = model.storageOutPowerCap
            for t in time:
                dual_e = model.dual[constr1e[t,i]]
                if i in model.BATTERY_PLANTS:
                    dual_p = model.dual[constr1p[t,i]] \
                            + model.dual[constr2p[t,i]]
                else:
                    dual_p = model.dual[constr1p[t,i]]
                prod_coeff.loc[t,e_id] = dual_e
                prod_coeff.loc[t,p_id] = dual_p
        for i in model.NEW_BRANCHES:
            constr_u = model.newBranchFlowUpperLimit
            constr_l = model.newBranchFlowLowerLimit
            for t in time:
                dual = model.dual[constr_u[t,i]] \
                        - model.dual[constr_l[t,i]]
                prod_coeff.loc[t,i[0]] = dual
            
        self.prod_coeff_sum = prod_coeff.sum()
        
        UNITS = list(model.POWER_PLANTS) + list(model.HYDROGEN_COMPONENTS) \
                    + list(model.BATTERY_COMPONENTS) + list(model.NEW_BRANCHES)
        
        for i in UNITS:
            if i in model.THERMAL_POWER_PLANTS:
                cap = model.Available_plants[i].value
            elif i in model.NEW_BRANCHES:
                cap = model.New_branch_cap[i].value
                i = i[0]
            else:
                cap = model.New_cap[i].value
            self.cut_const.loc[self.itr,i]  = cap*self.prod_coeff_sum[i]
            self.cut_grad.loc[self.itr,i]  =  self.prod_coeff_sum[i]
            
#        self.cut_const = self.cut_const.round(8)
#        self.cut_grad = self.cut_grad.round(8)
        
        invModel = self.invModelInstance
        
        NC_UNITS = list(model.RENEWABLE_POWER_PLANTS) \
                    + list(model.HYDROGEN_COMPONENTS) \
                    + list(model.BATTERY_COMPONENTS)
        
        invModel.cut.add(invModel.alpha \
                - sum(self.cut_grad.loc[self.itr,i]*invModel.available_plants[i] - self.cut_const.loc[self.itr,i] for i in model.THERMAL_POWER_PLANTS) \
                - sum(self.cut_grad.loc[self.itr,i[0]]*invModel.new_branch_cap[i] - self.cut_const.loc[self.itr,i[0]] for i in model.NEW_BRANCHES) \
                - sum(self.cut_grad.loc[self.itr,i]*invModel.new_cap[i] - self.cut_const.loc[self.itr,i] for i in NC_UNITS) \
                >= self.cut_w.loc[self.itr,'w'])
        
       # self.inv_opt.add_constraint(invModel.cut[self.itr])
        
    def updateOprModel(self):
        
        invModel = self.invModelInstance
        oprModel = self.oprModelInstance
        
        if hasattr(invModel,'initialRetiredCap'):
            invModel.del_component(invModel.initialRetiredCap)
#        if hasattr(invModel,'initialPlantCap'):
#            invModel.del_component(invModel.initialPlantCap)
#            
#        if hasattr(invModel,'hydrogenRatio') and self.itr > 3:
#            invModel.del_component(invModel.hydrogenRatio)
#        if hasattr(invModel,'batteryRatio') and self.itr > 3:
#            invModel.del_component(invModel.batteryRatio)
            
#        if self.itr > 15:
#            oprModel.branchFlow.activate()

        for i in  oprModel.RENEWABLE_POWER_PLANTS:
            oprModel.New_cap[i] = invModel.new_cap[i].value
                
        for i in oprModel.STORAGE_PLANTS:
            e_id = oprModel.STORAGE_ENERGY_AT_PLANT[i].value_list[0]
            p_id = oprModel.STORAGE_POWER_AT_PLANT[i].value_list[0]
            oprModel.New_cap[e_id] = invModel.new_cap[e_id].value
            oprModel.New_cap[p_id] = invModel.new_cap[p_id].value
            
        for i in oprModel.THERMAL_POWER_PLANTS:
            oprModel.Available_plants[i] = invModel.available_plants[i].value
                
        for i in oprModel.NEW_BRANCHES:
            oprModel.New_branch_cap[i] = invModel.new_branch_cap[i].value
        
        
def buildInvModel(mutables = {}):
    
    
    mutable_dict = {'inv_cost': False}
    
    for i in mutables.keys():
        mutable_dict[i] = mutables[i]
    
    m = pe.AbstractModel('detInvModel')
    
    m.BRANCHES = pe.Set(dimen = 3)
    m.NEW_BRANCHES = pe.Set(dimen = 3)
    
    m.PLANTS = pe.Set()
    m.SOLAR_POWER_PLANTS = pe.Set()
    m.WIND_POWER_PLANTS = pe.Set()  
    m.POWER_PLANTS = pe.Set()
    m.THERMAL_POWER_PLANTS = pe.Set()
    
    m.PLANT_TYPES = pe.Set()
    m.THERMAL_PLANT_TYPES = pe.Set()
    m.RENEWABLE_PLANT_TYPES = pe.Set()
    m.TYPE_TO_PLANTS = pe.Set(m.PLANT_TYPES)
    m.TYPE_TO_THERMAL_PLANTS = pe.Set(m.THERMAL_PLANT_TYPES)
    m.TYPE_TO_RENEWABLE_PLANTS = pe.Set(m.RENEWABLE_PLANT_TYPES)
    
#    m.HYDROGEN_PLANTS = pe.Set()
    m.ELECTROLYSIS = pe.Set()
#    m.H2_STORAGE = pe.Set()
#    m.HYDROGEN_COMPONENTS = pe.Set()
#    
#    m.BATTERY_PLANTS = pe.Set()
#    m.BATTERY_POWER = pe.Set()
    m.BATTERY_ENERGY = pe.Set()
#    m.BATTERY_COMPONENTS = pe.Set()
#    
#    m.COMPONENTS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
#    m.ELECTROLYSIS_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
#    m.STORAGE_AT_H2PLANT = pe.Set(m.HYDROGEN_PLANTS)
#    
#    m.COMPONENTS_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
#    m.POWER_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
#    m.ENERGY_AT_BATTERY = pe.Set(m.BATTERY_PLANTS)
    
    m.STORAGE_PLANTS = pe.Set()

    m.Period_ratio = pe.Param(within = pe.NonNegativeReals)
    m.Inv_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals, mutable = mutable_dict['inv_cost'])
    m.Fixed_cost = pe.Param(m.PLANT_TYPES, within = pe.NonNegativeReals)
    m.Retirement_cost = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
    
    m.Branch_cost = pe.Param(m.NEW_BRANCHES,within = pe.NonNegativeReals)

    m.Plant_size = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
    m.Max_num_plants = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
    m.Min_prod = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
    m.MaxElec = pe.Param(within = pe.NonNegativeReals)
    m.MaxBatteryPower = pe.Param(within = pe.NonNegativeReals)
    m.Alpha_lower = pe.Param(within = pe.Reals)
    
    m.Init_cap = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
    m.Solar_cap_pot = pe.Param(m.SOLAR_POWER_PLANTS, within = pe.NonNegativeReals)
    m.Wind_cap_inst = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
    m.Wind_cap_pot = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
    m.Trans_cap = pe.Param(m.BRANCHES,within = pe.NonNegativeReals)
    
    m.Initial_storage = pe.Param(m.STORAGE_PLANTS, within = pe.NonNegativeReals)

    m.new_cap = pe.Var(m.PLANTS, within = pe.NonNegativeReals)
    m.available_plants = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeIntegers)
    m.retired_cap = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
    m.new_branch_cap = pe.Var(m.NEW_BRANCHES, within = pe.Binary)
    
    m.alpha = pe.Var(within = pe.Reals)
    
    def numPlants_rule(m,i):
        return m.Plant_size[i]*m.available_plants[i] <= m.Init_cap[i] + m.new_cap[i] - m.retired_cap[i]
    m.numPlants = pe.Constraint(m.THERMAL_POWER_PLANTS, rule = numPlants_rule)
        
        
    def initialRetiredCap_rule(m,i):
        return m.retired_cap[i] == 0
    m.initialRetiredCap = pe.Constraint(m.THERMAL_POWER_PLANTS, rule = initialRetiredCap_rule)
    
#    def initialPlantCap_rule(m,i):
#        if i in m.THERMAL_POWER_PLANTS:
#            return pe.Constraint.Skip
#        elif i in m.SOLAR_POWER_PLANTS:
#            if m.Solar_cap_pot[i] > 10.0:
#                return m.new_cap[i] >= 10.0
#            else:
#                return pe.Constraint.Skip
#        elif i in m.WIND_POWER_PLANTS:
#            if m.Wind_cap_pot[i] > 10.0:
#                return m.new_cap[i] >= 10.0
#            else:
#                return pe.Constraint.Skip
#        else:
#            return m.new_cap[i] >= 10.0
#    m.initialPlantCap = pe.Constraint(m.PLANTS, rule = initialPlantCap_rule)
#    
#    def hydrogenRatio_rule(m,i):
#        p = m.ELECTROLYSIS_AT_H2PLANT[i]
#        e = m.STORAGE_AT_H2PLANT[i]
#        return 8*m.new_cap[p] == m.new_cap[e]
#    m.hydrogenRatio = pe.Constraint(m.HYDROGEN_PLANTS, rule = hydrogenRatio_rule)
#    
#    def batteryRatio_rule(m,i):
#        p = m.POWER_AT_BATTERY[i]
#        e = m.ENERGY_AT_BATTERY[i]
#        return 8*m.new_cap[p] == m.new_cap[e]
#    m.batteryRatio = pe.Constraint(m.BATTERY_PLANTS, rule = batteryRatio_rule)
    
    def maxPlant_rule(m,i):
        return m.available_plants[i] <= m.Max_num_plants[i]
    m.maxPlant = pe.Constraint(m.THERMAL_POWER_PLANTS, rule = maxPlant_rule)
    
    def maxSolarCap_rule(m,i):
        return m.new_cap[i]  <= m.Solar_cap_pot[i]
    m.maxSolarCap = pe.Constraint(m.SOLAR_POWER_PLANTS, rule = maxSolarCap_rule)
    
    def maxWindCap_rule(m,i):
        return m.new_cap[i]  <= m.Wind_cap_pot[i]
    m.maxWindCap = pe.Constraint(m.WIND_POWER_PLANTS, rule = maxWindCap_rule)
    
#    def maxBatteryEnergy_rule(m,i):
#        return m.new_cap[i]  <= m.MaxBatteryEnergy[i]
#    m.maxBatteryEnergy = pe.Constraint(m.BATTERY_ENERGY, rule = maxBatteryEnergy_rule)
#    
    def maxBatteryPower_rule(m,i):
        return m.new_cap[i]  <= m.MaxBatteryPower
    m.maxBatteryPower = pe.Constraint(m.BATTERY_ENERGY, rule = maxBatteryPower_rule)
    
    def maxElec_rule(m,i):
        return m.new_cap[i]  <= m.MaxElec
    m.maxElec = pe.Constraint(m.ELECTROLYSIS, rule = maxElec_rule)
    
    m.cut = pe.ConstraintList()
    
    def alphaLimit_rule(m):
        return m.alpha >= m.Alpha_lower
    m.alphaLimit = pe.Constraint(rule = alphaLimit_rule)
    
    
    def obj_rule(m):
            return  m.Period_ratio*(sum(sum(m.Inv_cost[j]*m.new_cap[i] for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)\
                    + sum(sum(m.Retirement_cost[j]*m.retired_cap[i] + m.Fixed_cost[j]*m.Plant_size[i]*m.available_plants[i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)\
                    + sum(sum(m.Fixed_cost[j]*m.new_cap[i] for i in m.TYPE_TO_RENEWABLE_PLANTS[j])for j in m.RENEWABLE_PLANT_TYPES)\
                    + sum(m.Branch_cost[i]*m.Trans_cap[i]*m.new_branch_cap[i] for i in m.NEW_BRANCHES)) + m.alpha

    m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
        
    return m

def buildOprModel(mutables = {}):
    
        mutable_dict = {'CO2_cost': False,
                        'H2_load_scaling': True}
        
        for i in mutables.keys():
            mutable_dict[i] = mutables[i]
        
        m = pe.AbstractModel('detOprModel')
        
        ##Sets##
        m.TIME = pe.Set(ordered = True)
        m.LAST_TIME = pe.Set(ordered = True)
        
        m.NODES = pe.Set(ordered = True)
        m.CURRENT_BRANCHES = pe.Set(dimen = 3)
        m.NEW_BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES = pe.Set(dimen = 3)
        m.BRANCHES_AT_NODE = pe.Set(m.NODES, dimen = 3)
        
        m.PLANT_TYPES = pe.Set()
        m.THERMAL_PLANT_TYPES = pe.Set()
        m.PLANTS = pe.Set()
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
#        m.STORAGE_AT_NODE = pe.Set(m.NODES)
        
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
        
        m.Rationing_cost = pe.Param(within = pe.NonNegativeReals)
        m.CO2_cost = pe.Param(within = pe.NonNegativeReals, mutable = mutable_dict['CO2_cost' ])
        
        m.Load = pe.Param(m.TIME, m.LOAD, within = pe.NonNegativeReals)
        m.H2_load = pe.Param(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        
        m.Emission_coef = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Var_cost = pe.Param(m.THERMAL_PLANT_TYPES, within = pe.NonNegativeReals)
        m.Ramp_rate = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Plant_size = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Min_prod = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.Init_cap = pe.Param(m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.Wind_cap_inst = pe.Param(m.WIND_POWER_PLANTS, within = pe.NonNegativeReals)
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
        m.Susceptance = pe.Param(m.BRANCHES,within = pe.Reals) # Non-Negative?
        m.Ref_power = pe.Param(within = pe.NonNegativeReals)
        m.Branch_dir_at_node = pe.Param(m.NODES,m.BRANCHES, within = pe.Integers)
                
        # Variables
        m.exp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        m.imp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
        
        m.prod = pe.Var(m.TIME, m.POWER_PLANTS, within = pe.NonNegativeReals)
        m.New_cap = pe.Param(m.PLANTS, within = pe.NonNegativeReals, mutable = True)
        m.Available_plants = pe.Param(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals, mutable = True)
        m.retired_cap = pe.Var(m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        m.cur = pe.Var(m.TIME, m.RENEWABLE_POWER_PLANTS, within = pe.NonNegativeReals)
        m.gen_state = pe.Var(m.TIME, m.THERMAL_POWER_PLANTS, within = pe.NonNegativeReals)
        
        m.hydrogen_direct = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)
        m.hydrogen_import_ccs = pe.Var(m.TIME, m.HYDROGEN_PLANTS, within = pe.NonNegativeReals)

        m.to_storage = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        m.from_storage = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        m.storage_level = pe.Var(m.TIME, m.STORAGE_PLANTS, within = pe.NonNegativeReals)
        
        m.rat = pe.Var(m.TIME, m.NODES, within = pe.NonNegativeReals)
        m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within = pe.Reals)
        m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
        m.New_branch_cap = pe.Param(m.NEW_BRANCHES, within = pe.Binary, mutable = True)
        
        
        ## Constraints##
        
        # THERMAL POWER PLANTS
        def genState_rule(m,t,i):
            return m.gen_state[t,i] <= m.Available_plants[i]
        m.genState = pe.Constraint(m.TIME, m.THERMAL_POWER_PLANTS,
                                   rule = genState_rule)
        
        def maxProd_rule(m,t,i):
            return m.prod[t,i]  <= m.Plant_size[i]*m.gen_state[t,i]
        m.maxProd = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS,
                                  rule = maxProd_rule)
        
        def minProd_rule(m,t,i):
            return m.prod[t,i]  >= m.Min_prod[i]*m.gen_state[t,i]
        m.minProd = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS,
                                  rule = minProd_rule)
                
        def rampUpLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t,i] - m.prod[t-1,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.gen_state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampUpLimit = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS,
                                      rule = rampUpLimit_rule)
        
        def rampDownLimit_rule(m,t,i):
            if pe.value(t) > 0:
                return m.prod[t-1,i] - m.prod[t,i] <= m.Ramp_rate[i]*m.Plant_size[i]*m.gen_state[t,i]
            else:
                return pe.Constraint.Skip
        m.rampDownLimit = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS,
                                        rule = rampDownLimit_rule)

        
        # WIND POWER
        def windBalance_rule(m,t,i):
#            if pe.value(m.Init_cap[i]) + pe.value(m.New_cap[i]) > 0:
            return m.prod[t,i] + m.cur[t,i] == m.Wind_profile_inst[t,i]*m.Init_cap[i] \
                    +  m.Wind_profile_pot[t,i]*m.New_cap[i]
#            else:
#                return m.prod[t,i] + m.cur[t,i] == 0.0
        m.windBalance = pe.Constraint(m.TIME,m.WIND_POWER_PLANTS,
                                      rule = windBalance_rule)   
        # SOLAR POWER
        def solarBalance_rule(m,t,i):
#            if pe.value(m.New_cap[i]) > 0:
            return m.prod[t,i] + m.cur[t,i] == \
                        m.Solar_profile[t,i]*(m.Init_cap[i] + m.New_cap[i])
#            else:
#                return m.prod[t,i] + m.cur[t,i] == 0.0       
        m.solarBalance = pe.Constraint(m.TIME,m.SOLAR_POWER_PLANTS,
                                      rule = solarBalance_rule) 
        
#       Storage plants       
        def storageBalance_rule(m,t,i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            if t == 0:
                return m.storage_level[t,i] == \
                            m.Initial_storage[i]*m.New_cap[j] \
                            + m.to_storage[t,i] - m.from_storage[t,i]
            else:
                return m.storage_level[t,i] == \
                            m.storage_level[t-1,i] \
                            + m.to_storage[t,i] - m.from_storage[t,i]
        m.storageBalance = pe.Constraint(m.TIME, m.STORAGE_PLANTS,
                                         rule = storageBalance_rule)
        
        def endStorage_rule(m,t,i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            return m.storage_level[t,i] == m.Initial_storage[i]*m.New_cap[j]
        m.endStorage = pe.Constraint(m.LAST_TIME, m.STORAGE_PLANTS,
                                     rule = endStorage_rule)
        
        def storageEnergyCap_rule(m, t, i):
            j = m.STORAGE_ENERGY_AT_PLANT[i]
            return m.storage_level[t,i] <= m.New_cap[j]
        m.storageEnergyCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS,
                                           rule = storageEnergyCap_rule)
        
        def storageInPowerCap_rule(m, t, i):
            j = m.STORAGE_POWER_AT_PLANT[i]
            if i in m.BATTERY_PLANTS:
                #return m.Battery_in_ratio*m.to_storage[t,i] <= m.New_cap[j]
                return m.to_storage[t,i] <= m.New_cap[j]
            elif i in m.HYDROGEN_PLANTS:
                return m.hydrogen_direct[t,i] \
                    + m.to_storage[t,i] <= m.New_cap[j]
        m.storageInPowerCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS,
                                            rule = storageInPowerCap_rule)

        def storageOutPowerCap_rule(m, t, i):
            j = m.STORAGE_POWER_AT_PLANT[i]
            if i in m.BATTERY_PLANTS:
                #return m.Battery_out_ratio*m.from_storage[t,i] <= m.New_cap[j]
                return m.from_storage[t,i] <= m.New_cap[j]
            else:
                return pe.Constraint.Skip
        m.storageOutPowerCap = pe.Constraint(m.TIME, m.STORAGE_PLANTS,
                                             rule = storageOutPowerCap_rule)
        
#        def storageOutPowerCap2_rule(m, t, i):
#            j = m.STORAGE_ENERGY_AT_PLANT[i]
#            if i in m.BATTERY_PLANTS:
#                #return m.Battery_out_ratio*m.from_storage[t,i] <= m.New_cap[j]
#                return m.from_storage[t,i] <= m.New_cap[j]
#            else:
#                return pe.Constraint.Skip
#        m.storageOutPowerCap2 = pe.Constraint(m.TIME, m.STORAGE_PLANTS,
#                                             rule = storageOutPowerCap2_rule)           
        
        def hydrogenBalance_rule(m,t,i):
            return m.hydrogen_direct[t,i] + m.from_storage[t,i] \
                    + m.hydrogen_import[t,i] + m.hydrogen_import_ccs[t,i] \
                    == m.H2_load[t,i]*m.H2_load_scaling
        m.hydrogenBalance = pe.Constraint(m.TIME, m.HYDROGEN_PLANTS,
                                          rule = hydrogenBalance_rule)
           
        
        # Energy balance
        def energyBalance_rule(m,t,i):
            return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
                        + m.H2_storage_eff*m.to_storage[t,j]
                        for j in m.H2PLANT_AT_NODE[i]) \
                        + sum(m.Battery_in_ratio*m.to_storage[t,j] \
                        - m.Battery_out_ratio*m.from_storage[t,j]
                        for j in m.BATTERY_AT_NODE[i])
#            if (i in m.H2PLANT_AT_NODE.keys()) and (i in m.BATTERY_AT_NODE.keys()):
#                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
#                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
#                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
#                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
#                        + m.H2_storage_eff*m.to_storage[t,j]
#                        for j in m.H2PLANT_AT_NODE[i]) \
#                        + sum(m.Battery_in_ratio*m.to_storage[t,j] \
#                        - m.Battery_out_ratio*m.from_storage[t,j]
#                        for j in m.BATTERY_AT_NODE[i])
#            elif  (i in m.H2PLANT_AT_NODE.keys()):
#                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
#                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
#                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
#                        + sum(m.H2_direct_eff*m.hydrogen_direct[t,j] \
#                        + m.H2_storage_eff*m.to_storage[t,j]
#                        for j in m.H2PLANT_AT_NODE[i])
#            elif (i in m.BATTERY_AT_NODE.keys()):
#                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
#                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
#                        == sum(m.Load[t,j] for j in m.LOAD_AT_NODE[i]) \
#                        + sum(m.Battery_in_ratio*m.to_storage[t,j] \
#                        - m.Battery_out_ratio*m.from_storage[t,j]
#                        for j in m.BATTERY_AT_NODE[i])
#            else:
#                return sum(m.prod[t,j] for j in m.GEN_AT_NODE[i]) \
#                        + m.rat[t,i] + m.imp[t,i] - m.exp[t,i] \
#                        == sum(m.Consumer_load[t,j] for j in m.LOAD_AT_NODE[i]) 

        m.energyBalance = pe.Constraint(m.TIME, m.NODES,
                                        rule = energyBalance_rule)
             
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
        m.branchFlowLimit = pe.Constraint(m.TIME, m.CURRENT_BRANCHES,
                                          rule = branchFlowLimit_rule )
        
        def newBranchFlowUpperLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Trans_cap[n, i,j])):
                return m.branch_flow[t,n,i,j] <= m.New_branch_cap[n,i,j]*m.Trans_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] <= 10000
        m.newBranchFlowUpperLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES,
                                                  rule = newBranchFlowUpperLimit_rule )
        
        def newBranchFlowLowerLimit_rule(m,t,n,i,j):
            if not np.isinf(pe.value(m.Trans_cap[n, i,j])):
                return m.branch_flow[t,n,i,j] >= -m.New_branch_cap[n,i,j]*m.Trans_cap[n,i,j] 
            else:
                return m.branch_flow[t,n,i,j] >= -10000
        m.newBranchFlowLowerLimit = pe.Constraint(m.TIME, m.NEW_BRANCHES,
                                                  rule = newBranchFlowLowerLimit_rule )
        
        def nodalBalance_rule(m,t,i):
            return m.imp[t,i] - m.exp[t,i] == m.Ref_power*sum(m.Branch_dir_at_node[i,j]*m.branch_flow[t,j] for j in m.BRANCHES_AT_NODE[i])
        m.nodalBalance = pe.Constraint(m.TIME, m.NODES, rule = nodalBalance_rule)   
        
        def obj_rule(m):
            return  sum(sum(sum((m.Var_cost[j] + m.Emission_coef[j]*m.CO2_cost)*m.prod[t,i] for i in m.TYPE_TO_THERMAL_PLANTS[j])for j in m.THERMAL_PLANT_TYPES)
                        + sum(m.Rationing_cost*m.rat[t,i] for i in m.NODES) \
                        + sum((m.Hydrogen_import_cost[i] + m.Hydrogen_CO2_emissions*m.CO2_cost)*m.hydrogen_import[t,i]
                            + (m.Hydrogen_import_cost_ccs[i] + m.Hydrogen_CO2_emissions_ccs*m.CO2_cost)*m.hydrogen_import_ccs[t,i] for i in m.HYDROGEN_PLANTS) for t in m.TIME)

        m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
        
        return m
    
def getBranchesAtNode(di):
    out = {}
    for node in di['NODES'][None]:
        for n,i,j in di['BRANCHES'][None]:
            if i == node or j == node:
                if node not in out.keys():
                    out[node] = []
                out[node].append((n,i,j))
    return out    
    
def addComponentStructureData(obj, di):
    
    # Nodes and branches
    
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
    
    
    di['BRANCHES_AT_NODE'] = getBranchesAtNode(di)
    
    # Plants and components
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('PlantType', inplace = True)
    
    h2_plant_char = copy.copy(obj.data.hydrogen_plant_char)
    h2_plant_char.set_index('Type', inplace = True)
    
    di['PLANT_TYPES'] = {None: plant_char.index.to_list() + h2_plant_char.index.to_list()}
    thermal_plants = plant_char.index[obj.data.plant_char['Variable cost ($/MWh)'] > 0]
    di['THERMAL_PLANT_TYPES'] = {None: thermal_plants.to_list()}
    di['RENEWABLE_PLANT_TYPES'] = {None: ['Wind','Solar']}
    
    obj.type2prefix = {'Biomass' : 'B', 'CC Gas' : 'CCG', 'CT Gas' : 'CTG',
                            'ICE Gas' : 'ICEG', 'CCS Gas' : 'CCSG',
                            'Coal' : 'C', 'CCS Coal' : 'CCSC', 'Nuclear' : 'N',
                            'Solar' : 'S', 'Wind' : 'W', 'Elec' : 'E',
                            'H2_Storage' : 'HS', 'Hydrogen': 'H', 'Load': 'L',
                            'H2_Load': 'H2L', 'Battery': 'ES',
                            'Battery Power': 'ESP', 'Battery Energy':'ESE'}
    
    obj.set2type = {'BIOMASS_POWER_PLANTS': 'Biomass',
                    'COAL_POWER_PLANTS': 'Coal',
                    'CCS_COAL_POWER_PLANTS': 'CCS Coal',
                    'CC_GAS_POWER_PLANTS': 'CC Gas',
                    'CT_GAS_POWER_PLANTS': 'CT Gas',
                    'ICE_GAS_POWER_PLANTS': 'ICE Gas',
                    'CCS_GAS_POWER_PLANTS': 'CCS Gas',
                    'NUCLEAR_POWER_PLANTS': 'Nuclear',
                    'SOLAR_POWER_PLANTS': 'Solar',
                    'WIND_POWER_PLANTS': 'Wind',
                    'HYDROGEN_PLANTS': 'Hydrogen',
                    'ELECTROLYSIS': 'Elec',
                    'HYDROGEN_STORAGE': 'H2_Storage',
                    'BATTERY_PLANTS': 'Battery',
                    'BATTERY_POWER': 'Battery Power', 
                    'BATTERY_ENERGY': 'Battery Energy'}
    
    for k in obj.set2type.keys():
        di[k] = {None: [obj.type2prefix[obj.set2type[k]] + '%.2d' % i for i in node_data.Bus.tolist()]}
        
#    wind_series = copy.copy(obj.data.wind_series)
#    di['WIND_POWER_PLANTS'] = {None: [obj.type2prefix['Wind'] + '%.2d' % int(i) for i in wind_series.columns.levels[0].tolist()]}
#    solar_series = copy.copy(obj.data.solar_series)
#    di['SOLAR_POWER_PLANTS'] = {None: [obj.type2prefix['Solar'] + '%.2d' % int(i) for i in solar_series.columns.tolist()]}   
    
    di['HYDROGEN_COMPONENTS'] = {None: di['ELECTROLYSIS'][None] \
                                  + di['HYDROGEN_STORAGE'][None]}
    
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
    
    obj.comp2node = {'GEN_AT_NODE':'POWER_PLANTS',
                     'LOAD_AT_NODE': 'LOAD',
                     'H2_LOAD_AT_NODE': 'H2_LOAD',
                     'H2PLANT_AT_NODE': 'HYDROGEN_PLANTS',
                     'BATTERY_AT_NODE': 'BATTERY_PLANTS'}
    
    for k in obj.comp2node.keys():
        di[k] = {i:[j for j in di[obj.comp2node[k]][None]
                    if int(j[-2:]) == i] for i in di['NODES'][None]}
    
#    di['STORAGE_AT_NODE'] = {**di['H2PLANT_AT_NODE'], **di['BATTERY_AT_NODE']}
    
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

    obj.meta_components = ['Hydrogen', 'Battery']

    di['TYPE_TO_PLANTS'] = {obj.set2type[i]: di[i][None]
        for i in obj.set2type.keys() if obj.set2type[i] not in obj.meta_components}
    
    obj.thermal_plants = ['Biomass', 'CC Gas', 'CT Gas','ICE Gas', 'CCS Gas',
                          'Coal', 'CCS Coal', 'Nuclear']
    obj.renewable_plants = ['Wind', 'Solar']
    
    di['TYPE_TO_THERMAL_PLANTS'] = {obj.set2type[i]: di[i][None]
        for i in obj.set2type.keys() if obj.set2type[i] in obj.thermal_plants}
    
    di['TYPE_TO_RENEWABLE_PLANTS'] = {obj.set2type[i]: di[i][None]
        for i in obj.set2type.keys() if obj.set2type[i] in obj.renewable_plants}
    
    di['PLANTS'] = {None: di['POWER_PLANTS'][None]\
                          + di['HYDROGEN_COMPONENTS'][None]
                          + di['BATTERY_COMPONENTS'][None]}
    
    return di

def invData(obj):
    
    
    GW2MW = 1000
    KW2MW = 0.001
    
    di = {}
    
    di['Period_ratio'] = {None: len(obj.timerange)/8760}
    di['Alpha_lower'] = {None: -1E5}
    di['MaxElec'] = {None: 1E5}
    di['MaxBatteryPower'] = {None: 1E5}
    
    di = addComponentStructureData(obj, di)
    
    node_data = copy.copy(obj.data.bus)
    
    solar_cap = copy.copy(obj.data.solar_cap)
    solar_cap.index = [ 'S%.2d' % i for i in obj.data.solar_cap.Bus.tolist()]  
    di['Solar_cap_pot'] = solar_cap.Pot_cap.to_dict()
    
    wind_cap = obj.data.wind_cap
    for i in node_data.Bus.tolist():
        if i not in wind_cap['Bus'].tolist():
            wind_cap.loc[i,'Inst_cap'] = 0.0
            wind_cap.loc[i,'Pot_cap'] = 0.0
            wind_cap.loc[i,'Bus'] = int(i)
    
    wind_cap.index = [ 'W%.2d' % i for i in wind_cap.Bus.tolist()]
    wind_cap.sort_index(inplace = True)
    wind_cap['Bus'] = [int(i) for i in wind_cap.Bus]
    
    wind_cap.fillna(0, inplace = True)
    
    di['Wind_cap_inst'] = wind_cap.Inst_cap.to_dict()
    di['Wind_cap_pot'] = wind_cap.Pot_cap.to_dict()
    
    installed = copy.copy(obj.data.installed)
    installed.set_index('Bus', inplace = True)
    init_cap = installed
    init_cap.fillna(0, inplace = True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (obj.type2prefix[j],i) : init_cap[i,j] for i,j in init_cap.keys()}
    di['Init_cap'] = init_cap_dict
    
    max_num_plants = copy.copy(obj.data.max_num_plants)
    max_num_plants.set_index('Type', inplace = True)
    max_plants = {}
    for t in max_num_plants.index: 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            max_plants[p] = max_num_plants.plants.loc[t]
    di['Max_num_plants'] = max_plants # MW
    
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('PlantType', inplace = True)
    
    plant_size = plant_char['Typical Plant Size (MW)']
    p_size = {}
    for t in di['TYPE_TO_THERMAL_PLANTS'].keys(): 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            p_size[p] = plant_size.loc[t]
    di['Plant_size'] = p_size # MW
    
    h2_plant_char = copy.copy(obj.data.hydrogen_plant_char)
    h2_plant_char.set_index('Type', inplace = True)
    
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
    
    retirement_cost = copy.copy(obj.data.retirement_cost)
    retirement_cost.index = obj.data.retirement_cost.Type
    di['Retirement_cost'] = retirement_cost.Cost.to_dict()
    
    line_data = copy.copy(obj.data.line)
    line_data.index = list(zip(line_data.index, line_data.From,line_data.To))
    di['Branch_cost'] = line_data[line_data.Type == 'New'].Cost.to_dict()
    di['Trans_cap'] = line_data.Cap.to_dict()
    

    return {'invData':di}

def oprData(obj):
    
    
    GW2MW = 1000
    KW2MW = 0.001
        
    di = {}
    ##Set##
    di['TIME'] = {None: list(obj.timerange)}
    #di['TIME'] = {None: list(range(3))}
    di['LAST_TIME'] = {None: [list(obj.timerange)[-1]]}
    
    di = addComponentStructureData(obj, di)

    installed = copy.copy(obj.data.installed)
    
    solar_cap = copy.copy(obj.data.solar_cap)
    solar_cap.index = [ 'S%.2d' % i for i in obj.data.solar_cap.Bus.tolist()]
   
    wind_cap = obj.data.wind_cap    
    wind_cap.index = [ 'W%.2d' % i for i in wind_cap.Bus.tolist()]
    wind_cap.sort_index(inplace = True)
    wind_cap['Bus'] = [int(i) for i in wind_cap.Bus]
    
    load_series = copy.copy(obj.data.load_series)
    load_series.columns = [obj.type2prefix['Load'] + '%.2d' % int(i) for i in load_series.columns]
    load_series = load_series[load_series.index.isin(obj.time)]
    load_series.index = list(obj.timerange)
    
    ##Parameters##
    di['NTime'] = {None: len(obj.timerange)}
    
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
    
    
    installed = copy.copy(obj.data.installed)
    installed.set_index('Bus', inplace = True)
    init_cap = installed
    init_cap.fillna(0, inplace = True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (obj.type2prefix[j],i) : init_cap[i,j] for i,j in init_cap.keys()}
    di['Init_cap'] = init_cap_dict
    
    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('PlantType', inplace = True)
    
    plant_size = plant_char['Typical Plant Size (MW)']
    p_size = {}
    a_plants = {}
    for t in di['TYPE_TO_THERMAL_PLANTS'].keys(): 
        for p in di['TYPE_TO_THERMAL_PLANTS'][t]:
            p_size[p] = plant_size.loc[t]
            a_plants[p] = np.round(init_cap_dict[p]/plant_size.loc[t])
    di['Plant_size'] = p_size # MW
    di['Available_plants'] = a_plants
    
    new_cap = {}
    for i in di['PLANTS'][None]:
         new_cap[i] = 0.0
    di['New_cap'] = new_cap  
        
    
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
    
    var_cost = plant_char['Variable cost ($/MWh)']
    var_cost = var_cost[var_cost.index.isin(di['THERMAL_PLANT_TYPES'][None])]
    var_cost_dict = var_cost.to_dict()
#    h2_var_cost = h2_plant_char['Variable Costs [$/kg]']
#    var_cost_dict.update(h2_var_cost)
    di['Var_cost'] = var_cost_dict
    
    
    emission_coef = copy.copy(plant_char['Emission (kg/MWh)'])
    emission_coef = emission_coef[emission_coef.index.isin(di['THERMAL_PLANT_TYPES'][None])]
    di['Emission_coef'] = emission_coef.to_dict() # kg CO2/MWh
    
    di['Solar_cap_pot'] = solar_cap.Pot_cap.to_dict()
    
    solar_series = copy.copy(obj.data.solar_series)
    solar_series = solar_series[solar_series.index.isin(obj.time)]
    solar_series.index = pd.Index(np.arange(len(solar_series.index)))
    solar_series.rename(columns = {i : obj.type2prefix['Solar'] + '%.2d' % int(i) for i in solar_series.columns.tolist()},
                                  level = 0, inplace = True)
    
    
    solar_profile = solar_series
    for i in di['SOLAR_POWER_PLANTS'][None]:
        if i not in solar_profile.columns:
            solar_profile.loc[:,i] = 0.0
    di['Solar_profile'] = solar_profile.round(4).stack(level = 0).to_dict()
    
    wind_cap.fillna(0, inplace = True)
    di['Wind_cap_inst'] = wind_cap.Inst_cap.to_dict()
    di['Wind_cap_pot'] = wind_cap.Pot_cap.to_dict()
    
    h2_plant_char = copy.copy(obj.data.hydrogen_plant_char)
    h2_plant_char.set_index('Type', inplace = True)
    di['H2_storage_eff'] = {None: h2_plant_char.loc['Elec','Energy rate [MWh/kg]'] + 
                              h2_plant_char.loc['H2_Storage','Energy rate [MWh/kg]']} # MWh/Nm^3        
    di['H2_direct_eff'] = {None: h2_plant_char.loc['Elec','Energy rate [MWh/kg]']} # MWh/Nm^3
    di['Hydrogen_CO2_emissions'] = {None: float(param.CO2_H2_imp.values[0])} # kg/Nm^3
    di['Hydrogen_CO2_emissions_ccs'] = {None: float(param.CO2_H2_imp_ccs.values[0])} # kg/Nm^3
    di['Initial_storage'] = {i: 0.5 for i in di['STORAGE_PLANTS'][None]}
    
    hydrogen_ng = copy.copy(obj.data.hydrogen_ng)
    hydrogen_ng.set_index('Plant', inplace = True)
    di['Hydrogen_import_cost'] = hydrogen_ng.H2_ng.to_dict()
    di['Hydrogen_import_cost_ccs'] = hydrogen_ng.H2_ng_ccs.to_dict()
    
    di['H2_load_scaling'] = {None: 1.0}
    
    di['Battery_in_ratio'] = {None: float(param.battery_in_ratio.values[0])}
    di['Battery_out_ratio'] = {None: float(param.battery_out_ratio.values[0])} 
    
    wind_series = copy.copy(obj.data.wind_series)
    wind_series = wind_series[wind_series.index.isin(obj.time)]
    wind_series.index = np.arange(len(wind_series.index))
    wind_series.rename(columns = {i : obj.type2prefix['Wind'] + '%.2d' % int(i) for i in wind_series.columns.levels[0].tolist()},
                                  level = 0, inplace = True)
    idx = pd.IndexSlice
    wind_profile_inst = wind_series.loc[idx[:],idx[:,'Inst_cap']]
    wind_profile_pot = wind_series.loc[idx[:],idx[:,'Pot_cap']]
    for i in di['WIND_POWER_PLANTS'][None]:
        if i not in wind_profile_inst.columns:
            wind_series.loc[:,(i,'Inst_cap')] = 0.0
        if i not in wind_profile_pot.columns:
            wind_series.loc[:,(i,'Pot_cap')] = 0.0
    
    di['Wind_profile_inst'] = wind_series.round(4).stack(level = 0).Inst_cap.fillna(0).to_dict()
    di['Wind_profile_pot'] = wind_series.round(4).stack(level = 0).Pot_cap.to_dict()
    
    line_data = copy.copy(obj.data.line)
    line_data.index = list(zip(line_data.index, line_data.From,line_data.To))
    di['Branch_cost'] = line_data[line_data.Type == 'New'].Cost.to_dict()
    di['Trans_cap'] = line_data.Cap.to_dict()
    di['Susceptance'] = line_data.B.to_dict()
    
    new_branch_cap = {}
    for i in di['Branch_cost'].keys():
        new_branch_cap[i] = 0.0
    di['New_branch_cap'] = new_branch_cap

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
    
    return {'oprData':di}
