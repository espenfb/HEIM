# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:04:29 2019

@author: espenfb
"""


import pyomo.environ as pe
import numpy as np


class elecInvModel(object):
    
    
    
    def __init__(self, price, hydrogen_param, print_text = False):
        
        self.price = price
        self.hydrogen_param = hydrogen_param
        self.time = np.arange(len(price)).tolist()
        
        if print_text:
            print('Building investment model...')
        self.model = self.buildModel()
    
        # Create concrete instance
        self.dataInstance = self.buildData()
        if print_text:
            print('Creating LP problem instance...')
        self.instance = self.model.create_instance(
                                data= self.dataInstance,
                                name="Electrolysis Investment Model",
                                namespace='Data')
        
        # Enable access to duals
        self.instance.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        
        
    def buildData(self):
        
        di = {}
        
        di['TIME'] = {None: self.time}

        di['Elec_cost'] = {None: self.hydrogen_param['Elec_cost']}
        di['Storage_cost'] = {None: self.hydrogen_param['Storage_cost']}
        
        di['Price'] = {t : self.price[t] for t in self.time}
        di['Import_cost'] = {None: self.hydrogen_param['Import_cost']}
        di['Elec_conv'] = {None: self.hydrogen_param['Elec_conv']}
        di['Storage_conv'] = {None: self.hydrogen_param['Storage_conv']}
        di['Hydrogen_demand'] = {t : self.hydrogen_param['Hydrogen_demand']
                                                            for t in self.time}
        di['Initial_storage'] = {None: self.hydrogen_param['Initial_storage']}
        di['Elec_max'] = {None: self.hydrogen_param['Elec_max']}
        di['Storage_max'] = {None: self.hydrogen_param['Storage_max']}
        
        return {'Data': di}
        
    def buildModel(self):
        
        
        m = pe.AbstractModel('Electrolysis Investment Model')
        
        
#        m.ELECTROLYSER
#        m.STORAGE
        m.TIME = pe.Set(within = pe.NonNegativeIntegers, ordered = True)
        
        m.Elec_cost = pe.Param(within = pe.NonNegativeReals)
        m.Storage_cost = pe.Param(within = pe.NonNegativeReals)
        
        m.Price = pe.Param(m.TIME, within = pe.Reals)
        m.Import_cost = pe.Param(within = pe.NonNegativeReals)
        m.Elec_conv = pe.Param(within = pe.NonNegativeReals)
        m.Storage_conv = pe.Param(within = pe.NonNegativeReals)
        m.Hydrogen_demand = pe.Param(m.TIME, within = pe.NonNegativeReals)
        m.Initial_storage = pe.Param(within = pe.NonNegativeReals)
        m.Elec_max = pe.Param(within = pe.NonNegativeReals)
        m.Storage_max = pe.Param(within = pe.NonNegativeReals)
        
        m.elec_cap = pe.Var(within = pe.NonNegativeReals)
        m.storage_cap = pe.Var(within = pe.NonNegativeReals)
        
        m.hydrogen_direct = pe.Var(m.TIME, within = pe.NonNegativeReals)
        m.hydrogen_to_storage = pe.Var(m.TIME, within = pe.NonNegativeReals)
        m.hydrogen_from_storage = pe.Var(m.TIME, within = pe.NonNegativeReals)
        m.hydrogen_to_load = pe.Var(m.TIME, within = pe.NonNegativeReals)
        m.hydrogen_import = pe.Var(m.TIME, within = pe.NonNegativeReals)
        m.storage_level = pe.Var(m.TIME, within = pe.NonNegativeReals)
        
        
        
        def obj_rule(m):
            return m.Elec_cost*m.elec_cap + m.Storage_cost*m.storage_cap \
                   + sum(m.Price[t]*(m.hydrogen_direct[t]*m.Elec_conv \
                       + m.hydrogen_to_storage[t]*m.Storage_conv) \
                       + m.hydrogen_import[t]*m.Import_cost for t in m.TIME)
        m.obj = pe.Objective(rule = obj_rule, sense = pe.minimize)
            
        def elecCap_rule(m,t):
            return m.Elec_conv*(m.hydrogen_direct[t] \
                                + m.hydrogen_to_storage[t]) <= m.elec_cap
        m.elecCap = pe.Constraint(m.TIME, rule = elecCap_rule)
        
        def elecCapMax_rule(m,t):
            return m.elec_cap <= m.Elec_max
        m.elecCapMax = pe.Constraint(m.TIME, rule = elecCapMax_rule)
        
#        def pumpCapMax_rule(m,t):
#            return  m.hydrogen_to_storage[t] <= m.elec_cap
#        m.pumpCapMax = pe.Constraint(m.TIME, rule = pumpCapMax_rule)
            
        def storageCap_rule(m,t):
            return m.storage_level[t] <= m.storage_cap
        m.storageCap = pe.Constraint(m.TIME, rule = storageCap_rule)
        
        def storageCapMax_rule(m,t):
            return m.storage_cap <= m.Storage_max
        m.storageCapMax = pe.Constraint(m.TIME, rule = storageCapMax_rule)
            
        def hydrogenBal_rule(m,t):
            return m.hydrogen_direct[t] + m.hydrogen_from_storage[t] \
                     + m.hydrogen_import[t] == m.Hydrogen_demand[t] 
        m.hydrogenBal = pe.Constraint(m.TIME, rule = hydrogenBal_rule)
        
        def storageBal_rule(m,t):
            if t == m.TIME[1]:
                return m.storage_level[t] == m.Initial_storage*m.storage_cap + \
                   m.hydrogen_to_storage[t] - m.hydrogen_from_storage[t]
            elif t == m.TIME[-1]:
                return m.storage_level[t] == m.Initial_storage*m.storage_cap
            else:
                return m.storage_level[t] == m.storage_level[t-1] + \
                       m.hydrogen_to_storage[t] - m.hydrogen_from_storage[t]
        m.storageBal = pe.Constraint(m.TIME, rule = storageBal_rule)
        
        return m
        
    def solve(self, printOutput = False, print_text = False):
        
        # Connect to solver
        opt = pe.SolverFactory('gurobi', solver_io='python')
    
        if print_text:
            print('Solving elec investment model...')
                
        # Solve model
        self.res = opt.solve(self.instance,
                        tee=printOutput, #stream the solver output
                        keepfiles=False, #print the LP file for examination
                        symbolic_solver_labels=True)





