import pyomo.environ as pe
import numpy as np


def buildDetModel(mutables={}):
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

    # SETS

    m.TIME = pe.Set(ordered=True)
    m.LAST_TIME = pe.Set(ordered=True)

    m.NODES = pe.Set()
    m.EL_NODES = pe.Set()
    m.H2_NODES = pe.Set()
    m.INTERNAL_NODES = pe.Set()
    m.MARKET_NODES = pe.Set()

    m.CURRENT_BRANCHES = pe.Set(dimen=3)
    m.NEW_BRANCHES = pe.Set(dimen=3)
    m.BRANCHES = pe.Set(dimen=3)
    m.BRANCHES_AT_NODE = pe.Set(m.NODES, dimen=3)

    m.CURRENT_LINES = pe.Set(dimen=3)
    m.NEW_LINES = pe.Set(dimen=3)
    m.LINES = pe.Set(dimen=3)
    # m.LINES_AT_NODE = pe.Set(m.NODES, dimen=3)

    m.PIPES = pe.Set(dimen=3)
    # m.PIPES_AT_NODE = pe.Set(m.NODES, dimen=3)

    m.PLANT_TYPES = pe.Set()
    m.THERMAL_PLANT_TYPES = pe.Set()
    m.RENEWABLE_PLANT_TYPES = pe.Set()
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

    m.GEN_AT_NODE = pe.Set(m.INTERNAL_NODES)
    m.CONV_AT_NODE = pe.Set(m.INTERNAL_NODES)
    m.STORAGE_AT_NODE = pe.Set(m.INTERNAL_NODES)
    m.AUX_POWER_AT_NODE = pe.Set(m.INTERNAL_NODES)

    m.TYPE_TO_PLANTS = pe.Set(m.PLANT_TYPES)
    m.TYPE_TO_THERMAL_PLANTS = pe.Set(m.THERMAL_PLANT_TYPES)
    m.TYPE_TO_STORAGE = pe.Set(m.STORAGE_TYPES)

    # Parameters
    m.Period_ratio = pe.Param(within=pe.NonNegativeReals)

    m.Rationing_cost = pe.Param(within=pe.NonNegativeReals)
    m.CO2_cost = pe.Param(within=pe.NonNegativeReals,
                          mutable=mutable_dict['CO2_cost'])
    m.Fuel_price = pe.Param(within=pe.NonNegativeReals,
                            mutable=mutable_dict['Fuel_price'])
    m.CCS_cost = pe.Param(within=pe.NonNegativeReals,
                          mutable=mutable_dict['CCS_cost'])
    m.Export_price = pe.Param(m.TIME, m.MARKET_NODES,
                              within=pe.Reals,
                              default=0)
    m.Import_price = pe.Param(m.TIME, m.MARKET_NODES,
                              within=pe.Reals,
                              default=0)

    m.Load = pe.Param(m.TIME, m.INTERNAL_NODES, within=pe.NonNegativeReals,
                      default=0)

    m.CO2_coef = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals)
    m.Power_cost = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals,
                            mutable=mutable_dict['Power_cost'])
    m.Energy_cost = pe.Param(m.STORAGE_TYPES, within=pe.NonNegativeReals,
                             mutable=mutable_dict['Energy_cost'])
    m.Fixed_energy_cost = pe.Param(
            m.STORAGE_TYPES, within=pe.NonNegativeReals)
    m.Fixed_power_cost = pe.Param(
            m.PLANT_TYPES, within=pe.NonNegativeReals)
    m.Var_cost = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals)
    m.Retirement_cost = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals,
                                 default=0)
    m.Retirement_rate = pe.Param(within=pe.NonNegativeReals, default=0.1)
    m.Fuel_rate = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals,
                           default=0)
    m.CCS_rate = pe.Param(m.PLANT_TYPES, within=pe.NonNegativeReals,
                          default=0)

    m.Ramp_rate = pe.Param(m.GEN_PLANTS, within=pe.NonNegativeReals,
                           default=1)
    m.Max_num_plants = pe.Param(m.GEN_PLANTS, within=pe.NonNegativeReals,
                                default=100)
    m.Plant_size = pe.Param(m.GEN_PLANTS, within=pe.NonNegativeReals,
                            default=100)
    m.Min_prod = pe.Param(m.GEN_PLANTS, within=pe.NonNegativeReals,
                          default=0)

    m.Init_power = pe.Param(m.PLANTS, within=pe.NonNegativeReals,
                            default=0)
    m.Init_energy = pe.Param(m.STORAGE, within=pe.NonNegativeReals,
                             default=0)
    m.Renewable_pot = pe.Param(m.RENEWABLE_POWER_PLANTS,
                               within=pe.NonNegativeReals,
                               default=0)
    m.Energy_max = pe.Param(m.STORAGE, within=pe.NonNegativeReals)

    m.Renewable_profile = pe.Param(m.TIME, m.RENEWABLE_POWER_PLANTS,
                                   within=pe.NonNegativeReals,
                                   default=0)
    #m.Inst_profile = pe.Param(m.TIME, m.ONSHORE_WIND_POWER_PLANTS,
    #                          within=pe.NonNegativeReals,
    #                          default=0)
    m.Inflow = pe.Param(m.TIME, m.STORAGE, within=pe.NonNegativeReals,
                        default=0)
    m.Inflow_ureg = pe.Param(
        m.TIME, m.STORAGE, within=pe.NonNegativeReals, default=0)
    m.Conv_rate = pe.Param(m.CONV_PLANTS, within=pe.NonNegativeReals)
    m.Aux_rate = pe.Param(m.STORAGE, within=pe.NonNegativeReals)

    m.Load_scaling = pe.Param(m.NODES, within=pe.NonNegativeReals, default=1,
                              mutable=mutable_dict['H2_load_scaling'])

    m.Init_storage = pe.Param(m.STORAGE, within=pe.NonNegativeReals,
                                 default=0)

    m.Eff_in = pe.Param(m.STORAGE, within=pe.NonNegativeReals)
    m.Eff_out = pe.Param(m.STORAGE, within=pe.NonNegativeReals)

    m.Branch_cap = pe.Param(m.BRANCHES, within=pe.NonNegativeReals)
    m.Branch_cost = pe.Param(m.NEW_BRANCHES, within=pe.NonNegativeReals)
    #m.Susceptance = pe.Param(m.BRANCHES,within = pe.Reals) # Non-Negative?
    m.Branch_dir_at_node = pe.Param(m.NODES, m.BRANCHES, within=pe.Integers)

    #m.Pipe_cap = pe.Param(m.PIPES,within = pe.NonNegativeReals)
    #m.Pipe_cost = pe.Param(m.NEW_PIPES,within = pe.NonNegativeReals)
    #m.Pipe_dir_at_node = pe.Param(m.NODES,m.PIPES, within = pe.Integers)

    # Variables
    m.exp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)
    m.imp = pe.Var(m.TIME, m.NODES, within=pe.NonNegativeReals)

    m.new_power = pe.Var(m.PLANTS, within=pe.NonNegativeReals)
    m.new_energy = pe.Var(m.STORAGE, within=pe.NonNegativeReals)

    m.available_plants = pe.Var(m.GEN_PLANTS, within=pe.NonNegativeReals)
    m.retired_cap = pe.Var(m.GEN_PLANTS, within=pe.NonNegativeReals)

    m.prod = pe.Var(m.TIME, m.PLANTS, within=pe.NonNegativeReals)
    m.cur = pe.Var(m.TIME, m.RENEWABLE_POWER_PLANTS,
                   within=pe.NonNegativeReals)
    m.state = pe.Var(m.TIME, m.GEN_PLANTS, within=pe.NonNegativeReals)

    m.to_storage = pe.Var(m.TIME, m.STORAGE, within=pe.NonNegativeReals)
    m.from_storage = pe.Var(m.TIME, m.STORAGE, within=pe.NonNegativeReals)
    m.storage = pe.Var(m.TIME, m.STORAGE, within=pe.NonNegativeReals)
    m.spill = pe.Var(m.TIME, m.STORAGE, within=pe.NonNegativeReals)

    m.rat = pe.Var(m.TIME, m.INTERNAL_NODES, within=pe.NonNegativeReals)
    m.branch_flow = pe.Var(m.TIME, m.BRANCHES, within=pe.Reals)
    # m.voltage_angle = pe.Var(m.TIME, m.NODES, within = pe.Reals)
    m.new_branch = pe.Var(m.NEW_BRANCHES, within=pe.Reals,
                          bounds=(0, 1))

    # Constraints

    # ALL POWER PLANTS

    def numPlants_rule(m, i):
        return m.Plant_size[i]*m.available_plants[i] <= m.Init_power[i] \
                 + m.new_power[i] - m.retired_cap[i]
    m.numPlants = pe.Constraint(m.GEN_PLANTS, rule=numPlants_rule)

    def genState_rule(m, t, i):
        return m.state[t, i] <= m.available_plants[i]
    m.genState = pe.Constraint(m.TIME, m.GEN_PLANTS, rule=genState_rule)

    def maxPlant_rule(m, i):
        return m.available_plants[i] <= m.Max_num_plants[i]
    m.maxPlant = pe.Constraint(m.GEN_PLANTS, rule=maxPlant_rule)

    def maxProd_rule(m, t, i):
        return m.prod[t, i] <= m.Plant_size[i]*m.state[t, i]
    m.maxProd = pe.Constraint(m.TIME, m.GEN_PLANTS, rule=maxProd_rule)

    def minProd_rule(m, t, i):
        return m.prod[t, i] >= m.Min_prod[i]*m.state[t, i]
    m.minProd = pe.Constraint(m.TIME, m.GEN_PLANTS, rule=minProd_rule)

    def rampUpLimit_rule(m, t, i):
        if pe.value(t) > 0:
            return m.prod[t, i] - m.prod[t-1, i] <= \
                m.Ramp_rate[i]*m.Plant_size[i]*m.state[t, i]
        else:
            return pe.Constraint.Skip
    m.rampUpLimit = pe.Constraint(m.TIME, m.GEN_PLANTS, rule=rampUpLimit_rule)

    def rampDownLimit_rule(m, t, i):
        if pe.value(t) > 0:
            return m.prod[t-1, i] - m.prod[t, i] <= \
                m.Ramp_rate[i]*m.Plant_size[i]*m.state[t, i]
        else:
            return pe.Constraint.Skip
    m.rampDownLimit = pe.Constraint(m.TIME, m.GEN_PLANTS,
                                    rule=rampDownLimit_rule)

    # RENEWABLES
    def renewableResourceLimit_rule(m, i):
        return m.new_power[i] <= m.Renewable_pot[i]
    m.renewableResourceLimit = pe.Constraint(
            m.RENEWABLE_POWER_PLANTS, rule=renewableResourceLimit_rule)

    def renewableBalance_rule(m, t, i):
        # if i in m.ONSHORE_WIND_POWER_PLANTS:
        #    return m.prod[t, i] + m.cur[t, i] == \
        #        m.Inst_profile[t, i]*m.Init_power[i] \
        #        + m.Renewable_profile[t, i]*m.new_power[i]     
        # else:
        return m.prod[t, i] + m.cur[t, i] == \
            m.Renewable_profile[t, i]*(m.Init_power[i] + m.new_power[i])
    m.renewableBalance = pe.Constraint(m.TIME, m.RENEWABLE_POWER_PLANTS,
                                       rule=renewableBalance_rule)

    #        def startUp_rule(m,t,i):
    #            return m.state[t,i] == m.state[t-1,i] + m.start_up[t,i] - m.shut_down[t,i]
    #        m.startUp = pe.Constraint(m.TIME,m.THERMAL_POWER_PLANTS, rule = startUp_rule)

    #       Storage plants
    # Storage loss occurs at the storage side and is represented by a
    # efficiency, eta = (1-loss) where eta^in = eat^out = sqrt(eta)

    def storageBalance_rule(m, t, i):
        if t == 0:
            return m.storage[t, i] == \
                m.Init_storage[i]*(m.Init_energy[i] + m.new_energy[i]) \
                + m.Eff_in[i]*m.to_storage[t, i] \
                - m.from_storage[t, i]/m.Eff_out[i] + m.Inflow[t, i] \
                + m.Inflow_ureg[t, i] - m.spill[t, i]
        else:
            return m.storage[t, i] == m.storage[t-1, i] \
                + m.Eff_in[i]*m.to_storage[t, i] \
                - m.from_storage[t, i]/m.Eff_out[i] \
                + m.Inflow[t, i] + m.Inflow_ureg[t, i] - m.spill[t, i]
    m.storageBalance = pe.Constraint(
            m.TIME, m.STORAGE, rule=storageBalance_rule)

    def endStorage_rule(m, t, i):
        return m.storage[t, i] == \
            m.Init_storage[i]*(m.Init_energy[i] + m.new_energy[i])
    m.endStorage = pe.Constraint(
            m.LAST_TIME, m.STORAGE, rule=endStorage_rule)

    def storageEnergyCap_rule(m, t, i):
        return m.storage[t, i] <= m.Init_energy[i] + m.new_energy[i]
    m.storageEnergyCap = pe.Constraint(
            m.TIME, m.STORAGE, rule=storageEnergyCap_rule)

    def storageInPowerCap_rule(m, t, i):
        return m.to_storage[t, i] <= m.Init_power[i] + m.new_power[i]
    m.storageInPowerCap = pe.Constraint(
            m.TIME, m.STORAGE, rule=storageInPowerCap_rule)

    def storageOutPowerCap_rule(m, t, i):
        return m.from_storage[t, i] <= m.Init_power[i] + m.new_power[i]
    m.storageOutPowerCap = pe.Constraint(
            m.TIME, m.STORAGE, rule=storageOutPowerCap_rule)

    # def maxStorageCap_rule(m,i):
    #     return m.new_energy[i]  <= m.Energy_max[i]
    # m.maxStorageCap = pe.Constraint(m.STORAGE, rule = maxStorageCap_rule)

    # HYDROPOWER
    def newHydroPower_rule(m, i):
        return m.new_power[i] == 0.0
    m.newHydroPower = pe.Constraint(
            m.HYDRO_STORAGE, rule=newHydroPower_rule)

    def newHydroEnergy_rule(m, i):
        return m.new_energy[i] == 0.0
    m.newHydroEnergy = pe.Constraint(
            m.HYDRO_STORAGE, rule=newHydroEnergy_rule)

    def minHydroProd_rule(m, t, i):
        return m.from_storage[t, i] + m.spill[t, i] >= m.Inflow_ureg[t, i]
    m.minHydroProd = pe.Constraint(
            m.TIME, m.HYDRO_STORAGE, rule=minHydroProd_rule)

    # NODES
    def energyBalance_rule(m, t, i):
        return sum(m.prod[t, j] for j in m.GEN_AT_NODE[i]) \
            + sum(m.from_storage[t, j] - m.to_storage[t, j]
                  for j in m.STORAGE_AT_NODE[i]) \
            + m.rat[t, i] + m.imp[t, i] - m.exp[t, i]\
            == m.Load[t, i]*m.Load_scaling[i] \
            + sum(m.Conv_rate[j]*m.prod[t, j] for j in m.CONV_AT_NODE[i]) \
            + sum(m.Aux_rate[j]*m.to_storage[t, j]
                  for j in m.AUX_POWER_AT_NODE[i])
    m.energyBalance = pe.Constraint(
            m.TIME, m.INTERNAL_NODES, rule=energyBalance_rule)

    def nodalBalance_rule(m, t, i):
        return m.imp[t, i] - m.exp[t, i] == \
            sum(m.Branch_dir_at_node[i, j]*m.branch_flow[t, j]
                for j in m.BRANCHES_AT_NODE[i])
    m.nodalBalance = pe.Constraint(m.TIME, m.NODES, rule=nodalBalance_rule)

    # BRANCHES

    # DC power flow
    #        def referenceNode_rule(m,t):
    #            return m.voltage_angle[t,m.NODES[1]] == 0.0
    #        m.ref_node = pe.Constraint(m.TIME, rule = referenceNode_rule)

    #        def branchFlow_rule(m,t,n,i,j):
    #            return m.branch_flow[t,n,i,j] == m.Susceptance[n,i,j]*(m.voltage_angle[t,i]-m.voltage_angle[t,j])
    #        m.branchFlow = pe.Constraint(m.TIME, m.BRANCHES, rule = branchFlow_rule)

    def branchFlowLimit_rule(m, t, n, i, j):
        if not np.isinf(m.Branch_cap[n, i, j]):
            return (-m.Branch_cap[n, i, j], m.branch_flow[t, n, i, j],
                    m.Branch_cap[n, i, j])
        else:
            return (-10000, m.branch_flow[t, n, i, j], 10000)
    m.branchFlowLimit = \
        pe.Constraint(m.TIME, m.CURRENT_LINES, rule=branchFlowLimit_rule)

    def newBranchFlowUpperLimit_rule(m, t, n, i, j):
        if not np.isinf(pe.value(m.Branch_cap[n, i, j])):
            return m.branch_flow[t, n, i, j] <= \
                m.new_branch[n, i, j]*m.Branch_cap[n, i, j] 
        else:
            return m.branch_flow[t, n, i, j] <= 10000
    m.newBranchFlowUpperLimit = \
        pe.Constraint(m.TIME, m.NEW_BRANCHES,
                      rule=newBranchFlowUpperLimit_rule)

    def newBranchFlowLowerLimit_rule(m, t, n, i, j):
        if not np.isinf(pe.value(m.Branch_cap[n, i, j])):
            return m.branch_flow[t, n, i, j] >= \
                -m.new_branch[n, i, j]*m.Branch_cap[n, i, j]
        else:
            return m.branch_flow[t, n, i, j] >= -10000
    m.newBranchFlowLowerLimit = \
        pe.Constraint(m.TIME, m.NEW_BRANCHES,
                      rule=newBranchFlowLowerLimit_rule)

    # OBJECTIVE
    # define cost components
    def plant_inv_cost(m):
        return sum(
                sum(m.Power_cost[j]*m.new_power[i]
                    for i in m.TYPE_TO_PLANTS[j]) for j in m.PLANT_TYPES)

    def storage_inv_cost(m):
        return sum(
                sum((m.Energy_cost[j] + m.Fixed_energy_cost[j])*m.new_energy[i]
                    + m.Fixed_power_cost[j] * m.new_power[i]
                    for i in m.TYPE_TO_STORAGE[j]) for j in m.STORAGE_TYPES)

    def thermal_fixed_and_retirement_costs(m):
        return sum(
                sum(m.Fixed_power_cost[j]*m.Plant_size[i]*m.available_plants[i]
                    + m.Retirement_rate*m.Power_cost[j]*m.retired_cap[i]
                    for i in m.TYPE_TO_THERMAL_PLANTS[j])
                for j in m.THERMAL_PLANT_TYPES)

    def branch_inv_cost(m):
        return sum(m.Branch_cost[i]*m.new_branch[i] for i in m.NEW_BRANCHES)

    def plant_opr_cost(m, t):
        return sum(
                sum((m.Fuel_price*m.Fuel_rate[j] + m.Var_cost[j]
                    + m.CO2_cost*m.CO2_coef[j]
                    + m.CCS_cost*m.CCS_rate[j])*m.prod[t, i]
                    for i in m.TYPE_TO_PLANTS[j])
                for j in m.PLANT_TYPES)

    def rationing_cost(m, t):
        return sum(m.Rationing_cost*m.rat[t, i] for i in m.INTERNAL_NODES)

    def market_cost(m, t):
        return sum(m.Import_price[t, i]*m.exp[t, i]
                   - m.Export_price[t, i]*m.imp[t, i] for i in m.MARKET_NODES)

    # add cost components to objective function

    def obj_rule(m):
        return m.Period_ratio*(plant_inv_cost(m)
                               + storage_inv_cost(m)
                               + branch_inv_cost(m)
                               + thermal_fixed_and_retirement_costs(m))\
               + sum(plant_opr_cost(m, t)
                     + rationing_cost(m, t)
                     + market_cost(m, t) for t in m.TIME)
    m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

    return m
