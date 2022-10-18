import copy
import numpy as np
import pandas as pd


def detData(obj):
    ''' Data input for the investment model according to the structure of the
    abstract model.'''
    di = {}
    # SET
    di['TIME'] = {None: list(obj.timerange)}
    di['LAST_TIME'] = {None: [list(obj.timerange)[-1]]}

    # Set node structure
    node_data = copy.copy(obj.data.bus)
    el_pf = 'EN'
    h2_pf = 'HN'
    market_pf = "MN"

    dual_buses = node_data[node_data.Type == "Dual"]

    di['EL_NODES'] = {None: [el_pf + '%.2d' %
                             i for i in dual_buses.Bus.tolist()]}
    di['H2_NODES'] = {None: [h2_pf + '%.2d' %
                             i for i in dual_buses.Bus.tolist()]}

    di['INTERNAL_NODES'] = {None: di['EL_NODES'][None] + di['H2_NODES'][None]}

    market_buses = node_data[node_data.Type == "Market"]
    market_buses_nr = market_buses.Bus.tolist()
    di['MARKET_NODES'] = {None: [market_pf + '%.2d' %
                                 i for i in market_buses_nr]}

    di['NODES'] = {None: di['INTERNAL_NODES'][None] + di['MARKET_NODES'][None]}

    def get_prefix(n, marked_buses, default=el_pf):
        if n in marked_buses:
            return market_pf
        else:
            return default

    # Set line structure
    line_data = copy.copy(obj.data.line)
    mb_str = ['%.2d' % i for i in market_buses_nr]
    branch_indx = []
    new_branch_indx = []
    pipe_indx = []
    for i in line_data.index:
        from_bus = '%.2d' % line_data.From[i]
        to_bus = '%.2d' % line_data.To[i]
        if line_data.Type[i] == 'Existing':
            from_typ = get_prefix(from_bus, mb_str)
            to_typ = get_prefix(to_bus, mb_str)
            branch_indx.append((i, from_typ + from_bus, to_typ + to_bus))
        elif line_data.Type[i] == 'New':
            from_typ = get_prefix(from_bus, mb_str)
            to_typ = get_prefix(to_bus, mb_str)
            new_branch_indx.append((i, from_typ + from_bus, to_typ + to_bus))
        elif line_data.Type[i] == 'H2':
            from_typ = get_prefix(from_bus, mb_str, default=h2_pf)
            to_typ = get_prefix(to_bus, mb_str, default=h2_pf)
            if (from_typ != market_pf) or (to_typ != market_pf):
                pipe_indx.append((i, from_typ + from_bus, to_typ + to_bus))

    di['CURRENT_LINES'] = {None: branch_indx}
    di['NEW_LINES'] = {None: new_branch_indx}
    di['LINES'] = {None: di['CURRENT_LINES'][None] + di['NEW_LINES'][None]}

    di['PIPES'] = {None: pipe_indx}

    di['BRANCHES'] = {None: di['LINES'][None] + di['PIPES'][None]}
    di['NEW_BRANCHES'] = {None: di['NEW_LINES'][None] + di['PIPES'][None]}

    def getBranchesAtNode(set_type):
        out = {}
        for node in di['NODES'][None]:
            for n, i, j in di[set_type][None]:
                if i == node or j == node:
                    if node not in out.keys():
                        out[node] = []
                    out[node].append((n, i, j))
        return out
    di['BRANCHES_AT_NODE'] = getBranchesAtNode('BRANCHES')

    # Define unit sets
    installed = copy.copy(obj.data.installed)
    renewable_pot = copy.copy(obj.data.renewable_pot)

    plant_char = copy.copy(obj.data.plant_char)
    plant_char.set_index('Type', inplace=True)
    plant_char.fillna(0, inplace=True)

    storage_char = copy.copy(obj.data.storage_char)
    storage_char.set_index('Type', inplace=True)
    storage_char.fillna(0, inplace=True)

    obj.type2prefix = {'Biomass': 'B', 'CC Gas': 'CCG', 'CT Gas': 'CTG',
                       'CCS Gas': 'CCSG', 'CT H2': 'CTH', 'CC H2': 'CCH',
                       'Coal': 'C', 'CCS Coal': 'CCSC', 'Nuclear': 'N',
                       'Solar': 'S', 'Offshore Wind': 'OW',
                       'Onshore Wind': 'SW',
                       'Hydrogen': 'HS', 'Battery': 'BS',
                       'PEMEL': 'PEMEL', 'PEMFC': 'PEMFC',
                       'SMR': 'SMR', 'SMR CCS': 'SMRCCS',
                       'Hydro': 'HP'} 
                     # 'ICE Gas' : 'ICEG','SOFC':'SOFC',
                     # 'ICE H2' : 'ICEH',

    di['type2prefix'] = obj.type2prefix

    #di['PLANT_TYPES'] = {
    #    None: plant_char.index.to_list() + storage_char.index.to_list()}
    di['PLANT_TYPES'] = {None: list(obj.type2prefix.keys())}
    #power_plants = plant_char.index[obj.data.plant_char.Class.isin(
    #    ['RE', 'TH', 'H2TH'])]
    #di['POWER_PLANT_TYPES'] = {None: power_plants.to_list()}
    di['POWER_PLANT_TYPES'] = {None: ['Biomass', 'CC Gas', 'CT Gas', 'CCS Gas',
                                      'Coal', 'CCS Coal', 'Nuclear'
                                      'CT H2', 'CC H2', 'PEMFC',
                                      'Solar', 'Offshore Wind', 'Onshore Wind',
                                      'Wind']}
    #thermal_plants = \
    #    plant_char.index[obj.data.plant_char.Class.isin(['TH', 'H2TH'])]
    #di['THERMAL_PLANT_TYPES'] = {None: thermal_plants.to_list()}
    di['THERMAL_PLANT_TYPES'] = {None: ['Biomass', 'CC Gas', 'CT Gas', 'CCS Gas',
                                        'CT H2', 'CC H2', 'PEMFC',
                                        'Coal', 'CCS Coal', 'Nuclear']}
    #renewable_plants = \
    #    plant_char.index[obj.data.plant_char.Class.isin(['RE'])]
    #di['RENEWABLE_PLANT_TYPES'] = {None: renewable_plants.to_list()}
    di['RENEWABLE_PLANT_TYPES'] = {None: ['Solar', 'Offshore Wind',
                                          'Onshore Wind']}
    #hydrogen_plants = \
    #    plant_char.index[obj.data.plant_char.Class.isin(['H2'])]
    #di['HYDROGEN_PLANT_TYPES'] = {None: hydrogen_plants.to_list()}
    di['HYDROGEN_PLANT_TYPES'] = {None: ['PEMEL', 'SMR', 'SMR CCS']}
    #di['STORAGE_TYPES'] = {None: storage_char.index.to_list()}
    di['STORAGE_TYPES'] = {None: ['Hydrogen', 'Battery', 'Hydro']}
    #h2_conv_plants = \
    #    plant_char.index[obj.data.plant_char.Class.isin(['H2TH'])]
    #di['CONV_TYPES'] = {None: ['PEMEL'] + h2_conv_plants.to_list()}
    di['CONV_TYPES'] = {None: ['PEMEL'] + ['CT H2', 'CC H2', 'PEMFC']}
    di['GAS_PLANT_TYPES'] = {None: ['CC Gas', 'CT Gas', 'CCS Gas']}
    di['SMR_PLANT_TYPES'] = {None: ['SMR', 'SMR CCS']}

    # Create individual plant set and set for plant-type to individual plants
    di['TYPE_TO_PLANTS'] = {}
    di['TYPE_TO_THERMAL_PLANTS'] = {}
    di['TYPE_TO_CONV_PLANTS'] = {}
    di['PLANTS'] = {None: []}
    obj.plant_buses_type = {}
    for k in obj.type2prefix.keys():
        if k in di['POWER_PLANT_TYPES'][None]:
            class_type = '_POWER_PLANTS'
        elif k in di['HYDROGEN_PLANT_TYPES'][None]:
            class_type = '_PLANTS'
        elif k in di['STORAGE_TYPES'][None]:
            class_type = '_STORAGE'

        set_name = k.replace(' ', '_').upper() + class_type
        if k in (['Hydro'] + di['RENEWABLE_PLANT_TYPES'][None]):
            # Plants that are created for only buses where cap exist
            #  or potential is explicitly defined
            exist = installed[k].notna()
            potential = renewable_pot[k].notna()
            to_include = pd.concat([exist, potential], axis=1).any(axis=1)
            obj.plant_buses_type[k] = installed["Bus"][to_include].values
            di[set_name] = {None: [obj.type2prefix[k] + '%.2d' %
                                   i for i in obj.plant_buses_type[k]]}
        elif k in plant_char.index.to_list():
            # Plants that are created for all dual buses,
            # potential is assumed everywhere
            di[set_name] = {None: [obj.type2prefix[k] + '%.2d' %
                                   i for i in dual_buses.Bus.tolist()]}
        else:
            di[set_name] = {None: []}
        di['PLANTS'][None] += [obj.type2prefix[k] + '%.2d' %
                               i for i in dual_buses.Bus.tolist()]

        if class_type == '_PLANTS':
            class_type = '_H2_PLANTS'
        if not 'TYPE_TO' + class_type in di.keys():
            di['TYPE_TO' + class_type] = {}
        di['TYPE_TO' + class_type][k] = di[set_name][None]
        di['TYPE_TO_PLANTS'][k] = di[set_name][None]
        if k in di['THERMAL_PLANT_TYPES'][None]:
            di['TYPE_TO_THERMAL_PLANTS'][k] = di[set_name][None]
        if k in di['CONV_TYPES'][None]:
            di['TYPE_TO_CONV_PLANTS'][k] = di[set_name][None]

    di['TYPE_TO_GEN_PLANTS'] = {
        **di['TYPE_TO_THERMAL_PLANTS'], **di['TYPE_TO_H2_PLANTS']}

    # -- COLLECTIONS (higher level sets/categories)--
    # All wind power plants
    di['WIND_POWER_PLANTS'] = {None: di['OFFSHORE_WIND_POWER_PLANTS'][None]
                               + di['ONSHORE_WIND_POWER_PLANTS'][None]}
    di['RENEWABLE_POWER_PLANTS'] = {None: di['WIND_POWER_PLANTS'][None]
                                    + di['SOLAR_POWER_PLANTS'][None]}
    # Storage of electricity only
    di['EL_STORAGE'] = {None: di['BATTERY_STORAGE']
                        [None] + di['HYDRO_STORAGE'][None]}
    # All storage
    di['STORAGE'] = {None: di['HYDROGEN_STORAGE']
                     [None] + di['EL_STORAGE'][None]}
    di['GAS_POWER_PLANTS'] = {None: di['CC_GAS_POWER_PLANTS'][None]
                              + di['CT_GAS_POWER_PLANTS'][None]
                              + di['CCS_GAS_POWER_PLANTS'][None]}
    # Plants generating el from h2
    di['H2_POWER_PLANTS'] = {None: di['CC_H2_POWER_PLANTS'][None]
                             + di['CT_H2_POWER_PLANTS'][None]
                             + di['PEMFC_POWER_PLANTS'][None]}
                            # + di['ICE_H2_POWER_PLANTS'][None]
                            # + di['SOFC_POWER_PLANTS'][None] }
    di['THERMAL_POWER_PLANTS'] = {None: di['BIOMASS_POWER_PLANTS'][None]
                                  + di['COAL_POWER_PLANTS'][None] +
                                  di['GAS_POWER_PLANTS'][None]
                                  + di['NUCLEAR_POWER_PLANTS'][None] +
                                  di['CCS_COAL_POWER_PLANTS'][None]
                                  + di['H2_POWER_PLANTS'][None]}
                                # + di['ICE_GAS_POWER_PLANTS'][None]}
    # Plants generating h2
    di['H2_PLANTS'] = {None: di['PEMEL_PLANTS'][None]
                       + di['SMR_PLANTS'][None] + di['SMR_CCS_PLANTS'][None]}
    # All plants generating el
    di['POWER_PLANTS'] = {None: di['RENEWABLE_POWER_PLANTS'][None]
                          + di['THERMAL_POWER_PLANTS'][None]}
    di['GEN_PLANTS'] = {None: di['H2_PLANTS'][None]
                        + di['THERMAL_POWER_PLANTS'][None]}
    di['CONV_PLANTS'] = {None: di['PEMEL_PLANTS']
                         [None] + di['H2_POWER_PLANTS'][None]}

    # -- Position of units in the networks --
    # Platns generating el at el nodes
    eln = di['EL_NODES'][None]
    h2n = di['H2_NODES'][None]
    n = di['NODES'][None]

    di['GEN_AT_NODE'] = {i: [j for j in di['POWER_PLANTS'][None]
                             if j[-2:] == i[-2:]] for i in eln}

    # Plants generating h2 at h2 node
    di['GEN_AT_NODE'].update({i: [j for j in (di['H2_PLANTS'][None])
                                  if j[-2:] == i[-2:]] for i in h2n})

    # Plants converting h2 into el at h2 nodes
    di['CONV_AT_NODE'] = {i: [j for j in (di['H2_POWER_PLANTS'][None])
                              if j[-2:] == i[-2:]] for i in h2n}
    # Plants converting el into h2 at el nodes
    di['CONV_AT_NODE'].update({i: [j for j in (di['PEMEL_PLANTS'][None])
                                   if j[-2:] == i[-2:]] for i in eln})

    di['BATTERY_STORAGE_AT_NODE'] = {i: [j for j in (di['BATTERY_STORAGE'][None])
                                         if j[-2:] == i[-2:]] for i in n}

    di['HYDRO_POWER_AT_NODE'] = {i: [j for j in (di['HYDRO_STORAGE'][None])
                                     if int(j[-2:]) == i] for i in n}

    # El storage at el nodes
    di['STORAGE_AT_NODE'] = {i: [j for j in (di['EL_STORAGE'][None])
                                 if j[-2:] == i[-2:]] for i in eln}
    # H2 storage at h2 nodes
    di['STORAGE_AT_NODE'].update({i: [j for j in (di['HYDROGEN_STORAGE'][None])
                                      if j[-2:] == i[-2:]] for i in h2n})

    di['AUX_POWER_AT_NODE'] = {i: [j for j in (di['HYDROGEN_STORAGE'][None])
                                   if j[-2:] == i[-2:]] for i in eln}

    # Parameters
    GW2MW = 1E3
    di['NTime'] = {None: len(obj.timerange)}
    di['Period_ratio'] = {None: len(obj.timerange)/8760}

    param = copy.copy(obj.data.parameters)

    load_series = copy.copy(obj.data.load_series)
    load_series.columns = [el_pf + '%.2d' %
                           int(i) for i in load_series.columns]
    load_series = load_series[load_series.index.isin(obj.time)]
    load_series.index = list(obj.timerange)

    di['Load'] = load_series.stack().to_dict()

    h2_load = pd.DataFrame(index=obj.data.load_series.index,
                           columns=di['H2_NODES'][None])
    hours_per_year = len(obj.data.load_series.index)
    scale = GW2MW/hours_per_year
    for i in di['H2_NODES'][None]:
        indx = obj.data.hydrogen_load.Bus == int(i[-2:])
        value = obj.data.hydrogen_load.loc[indx,
                                           'high'].values*scale  # MWh/h
        if len(value) > 0:
            h2_load.loc[:, i] = value[0]
    h2_load.fillna(0, inplace=True)
    h2_load = h2_load[h2_load.index.isin(obj.time)]
    h2_load.index = np.arange(len(h2_load.index))
    di['Load'].update(h2_load.stack().to_dict())

    export_price = copy.copy(obj.data.export_price_series)
    export_price.columns = [market_pf + '%.2d' %
                            int(i) for i in export_price.columns]
    export_price = export_price[export_price.index.isin(obj.time)]
    export_price.index = list(obj.timerange)
    di["Export_price"] = export_price.stack().to_dict()

    import_price = copy.copy(obj.data.import_price_series)
    import_price.columns = [market_pf + '%.2d' %
                            int(i) for i in import_price.columns]
    import_price = import_price[import_price.index.isin(obj.time)]
    import_price.index = list(obj.timerange)
    di["Import_price"] = import_price.stack().to_dict()

    installed.set_index('Bus', inplace=True)
    init_cap = installed
    init_cap.fillna(0, inplace=True)
    init_cap = init_cap.stack().to_dict()
    init_cap_dict = {'%s%.2d' % (
        obj.type2prefix[j], i): init_cap[i, j] for i, j in init_cap.keys()}
    di['Init_power'] = init_cap_dict

    init_energy = copy.copy(obj.data.installed_energy)
    init_energy.set_index('Bus', inplace=True)
    init_energy.fillna(0, inplace=True)
    init_energy = init_energy.stack().to_dict()
    init_energy_dict = {'%s%.2d' % (obj.type2prefix[j], i):
                        init_energy[i, j] for i, j in init_energy.keys()}
    init_energy_dict = {k: v*GW2MW for (k, v) in init_energy_dict.items()
                        if k in di["STORAGE"][None]}  # filter for storage items
    di['Init_energy'] = init_energy_dict

    ramp_rate = copy.copy(plant_char['Ramp (%/h)'])
    rate = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys():
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            rate[p] = ramp_rate.loc[t]
    di['Ramp_rate'] = rate  # %/h

    min_limit = copy.copy(plant_char['Min Gen (pu)'])
    m_lim = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys():
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            m_lim[p] = min_limit.loc[t]
    di['Min_prod'] = m_lim  # MW

    max_num_plants = copy.copy(obj.data.max_num_plants)
    max_num_plants.set_index('Type', inplace=True)
    max_plants = {}
    for t in max_num_plants.index:
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            max_plants[p] = max_num_plants.plants.loc[t]
    di['Max_num_plants'] = max_plants  # MW

    plant_size = plant_char['Typical Plant Size (pu)']
    p_size = {}
    for t in di['TYPE_TO_GEN_PLANTS'].keys():
        for p in di['TYPE_TO_GEN_PLANTS'][t]:
            p_size[p] = plant_size.loc[t]
    di['Plant_size'] = p_size  # MW

    Power_cost = plant_char['Inv ($/pu-year)']
    di['Power_cost'] = Power_cost.to_dict()
    di['Power_cost'].update(storage_char['Inv power ($/pu-yr)'].to_dict())

    di['Energy_cost'] = storage_char['Inv energy ($/eu-yr)'].to_dict()

    Fixed_power_cost = plant_char['Fix ($/pu-year)']
    di['Fixed_power_cost'] = Fixed_power_cost.to_dict()
    di['Fixed_power_cost'].update(
        storage_char['Fix power ($/pu-yr)'].to_dict())

    di['Fixed_energy_cost'] = storage_char['Fix energy ($/eu-yr)'].to_dict()

    var_cost = plant_char['Var ($/eu)']
    di['Var_cost'] = var_cost.to_dict()
    di['Var_cost'].update(storage_char['Var pow ($/eu)'].to_dict())

    fuel_rate = plant_char['Fuel (in eu/out eu)']
    # fuel_rate = fuel_rate[fuel_rate.index.isin(di['CONV_TYPES'][None])]
    fuel_rate_dict = fuel_rate.to_dict()
    di['Conv_rate'] = {}
    for k, v in di['TYPE_TO_CONV_PLANTS'].items():
        if k in fuel_rate_dict.keys():
            di['Conv_rate'].update({i: fuel_rate_dict[k] for i in v})
        else:
            di['Conv_rate'].update({i: 0.0 for i in v})

    di['Fuel_rate'] = {}
    fuel_types = di['GAS_PLANT_TYPES'][None] + di['SMR_PLANT_TYPES'][None]
    for i in fuel_types:
        if i in fuel_rate_dict.keys():
            di['Fuel_rate'].update({i: fuel_rate_dict[i]})
        else:
            di['Fuel_rate'].update({i: 0.0})

    aux_rate = storage_char['Aux power (MWh/eu)']
    aux_rate_dict = aux_rate.to_dict()
    di['Aux_rate'] = {}
    for k, v in di['TYPE_TO_STORAGE'].items():
        if k in aux_rate_dict.keys():
            di['Aux_rate'].update({i: aux_rate_dict[k] for i in v})
        else:
            di['Aux_rate'].update({i: 0.0 for i in v})

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
    solar_series.rename(columns={str(i): obj.type2prefix['Solar'] + '%.2d' % int(i)
                                 for i in obj.plant_buses_type["Solar"]},
                        level=0, inplace=True)
    solar_series[solar_series < 0] = 0.0
    include_plants = di["SOLAR_POWER_PLANTS"][None]
    solar_series = solar_series[include_plants]
    di['Renewable_profile'] = solar_series.round(4).stack(level=0).to_dict()

    wind_series = copy.copy(obj.data.wind_series)
    wind_series = wind_series[wind_series.index.isin(obj.time)]
    wind_series.index = np.arange(len(wind_series.index))
    wind_series.rename(columns={str(i): obj.type2prefix['Onshore Wind']
                       + '%.2d' % int(i)
                       for i in obj.plant_buses_type["Onshore Wind"]},
                       inplace=True)
    wind_series[wind_series < 0] = 0.0
    include_plants = di["ONSHORE_WIND_POWER_PLANTS"][None]
    wind_series = wind_series[include_plants]
    di['Renewable_profile'].update(wind_series.round(
        4).fillna(0).unstack().swaplevel().to_dict())

    offshore_wind_series = copy.copy(obj.data.offshore_wind_series)
    offshore_wind_series = \
        offshore_wind_series[offshore_wind_series.index.isin(obj.time)]
    offshore_wind_series.index = np.arange(len(offshore_wind_series.index))
    offshore_wind_series.rename(
        columns={str(i): obj.type2prefix['Offshore Wind'] + '%.2d' % int(i)
                 for i in obj.plant_buses_type["Offshore Wind"]},
        inplace=True)
    offshore_wind_series[offshore_wind_series < 0] = 0.0
    include_plants = di["OFFSHORE_WIND_POWER_PLANTS"][None]
    offshore_wind_series = offshore_wind_series[include_plants]
    di['Renewable_profile'].update(offshore_wind_series.round(
        4).fillna(0).unstack().swaplevel().to_dict())

    #inst_wind_series = copy.copy(obj.data.inst_wind_series)
    #inst_wind_series = inst_wind_series[inst_wind_series.index.isin(obj.time)]
    #inst_wind_series.index = np.arange(len(inst_wind_series.index))
    #inst_wind_series.rename(
    #    columns={i: obj.type2prefix['Onshore Wind'] + '%.2d' % int(i)
    #             for i in inst_wind_series.columns.tolist()},
    #    inplace=True)
    #inst_wind_series[inst_wind_series < 0] = 0.0
    #di['Inst_profile'] = inst_wind_series.round(
    #    4).fillna(0).unstack().swaplevel().to_dict()


    # Unregulated part of inflow
    if hasattr(obj.data, 'inflow_ureg_series'):
        inflow_ureg_series = copy.copy(obj.data.inflow_ureg_series)
        inflow_ureg_series.index = np.arange(len(inflow_ureg_series.index))
        inflow_ureg_series = inflow_ureg_series[inflow_ureg_series.index.isin(
            di['TIME'][None])]
        inflow_ureg_series.rename(
            columns={str(i): obj.type2prefix['Hydro'] + '%.2d' % int(i)
                     for i in obj.plant_buses_type["Hydro"]},
            inplace=True)
        inflow_ureg_series[inflow_ureg_series < 0] = 0.0
        include_plants = di["STORAGE"][None]
        for p in include_plants:
            if p not in inflow_ureg_series.columns:
                inflow_ureg_series.loc[:, p] = 0.0
        inflow_ureg_series = inflow_ureg_series[include_plants]
        di['Inflow_ureg'] = inflow_ureg_series.round(
            4).fillna(0).unstack().swaplevel().to_dict()
    else:
        di['Inflow_ureg'] = {}

    # Total inflow
    if hasattr(obj.data, 'inflow_series'):
        inflow_series = copy.copy(obj.data.inflow_series)

        inflow_series.index = np.arange(len(inflow_series.index))
        inflow_series = inflow_series[inflow_series.index.isin(
            di['TIME'][None])]
        inflow_series.rename(
            columns={str(i): obj.type2prefix['Hydro'] + '%.2d' % int(i)
                     for i in obj.plant_buses_type["Hydro"]},
            inplace=True)
        inflow_series[inflow_series < 0] = 0.0
        include_plants = di["STORAGE"][None]
        for p in include_plants:
            if p not in inflow_series.columns:
                inflow_series.loc[:, p] = 0.0
        inflow_series = inflow_series[include_plants]
        tot_inflow_series = inflow_series + inflow_ureg_series
        di['Inflow'] = tot_inflow_series.round(4).fillna(
            0).unstack().swaplevel().to_dict()
    else:
        di['Inflow'] = {}

    renewable_pot.set_index('Bus', inplace=True)
    #renewable_pot.fillna(0, inplace=True)
    renewable_pot = renewable_pot.stack().to_dict()
    renewable_pot_dict = {'%s%.2d' % (obj.type2prefix[j], i):
                          renewable_pot[i, j]
                          for i, j in renewable_pot.keys()}
    di['Renewable_pot'] = renewable_pot_dict

    eff_in = {}
    eff_out = {}
    en_max = {}
    for k, v in di['TYPE_TO_STORAGE'].items():
        for i in v:
            eff_in[i] = storage_char.loc[k, 'In (%)']
            eff_out[i] = storage_char.loc[k, 'Out (%)']
            en_max[i] = storage_char.loc[k, 'New Energy Max (eu)']

    di['Eff_in'] = eff_in
    di['Eff_out'] = eff_out
    di['Energy_max'] = en_max
    di['Init_storage'] = {i: 0.0 for i in di['STORAGE'][None]}
    di['Init_storage'].update({i: 0.6 for i in di["HYDRO_STORAGE"][None]})

    di['Load_scaling'] = {i: param['H2_scaling'].values[0]
                          for i in di['H2_NODES'][None]}

    new_indx = []
    for i in line_data.index:
        ltp = line_data.iloc[i].Type
        fn = '%.2d' % line_data.iloc[i].From
        tn = '%.2d' % line_data.iloc[i].To
        if ltp == 'H2':
            f_ntp = get_prefix(fn, mb_str, default=h2_pf)
            t_ntp = get_prefix(tn, mb_str, default=h2_pf)
            if (f_ntp == market_pf) | (t_ntp == market_pf):
                continue
        else:
            f_ntp = get_prefix(fn, mb_str)
            t_ntp = get_prefix(tn, mb_str)
        new_indx.append((i, f_ntp + fn, t_ntp + tn))

    line_data.index = new_indx
    di['Branch_cost'] = line_data[line_data.Type.isin(
        ['New', 'H2'])].Cost.to_dict()
    di['Branch_cap'] = line_data.Cap.to_dict()
    # di['Susceptance'] = line_data.B.to_dict()

    di['Rationing_cost'] = {None: param.at[0, 'rat_cost']}
    di['CO2_cost'] = {None: param.at[0, 'CO2_cost']}
    di['Fuel_price'] = {None: param.at[0, 'NG price ($/mmBtu)']}
    di['CCS_cost'] = {None: float(param.at[0, 'CCS cost ($/kg)'])}

    def getBranchDirAtNode():
        out = {}
        for node in di['NODES'][None]:
            for n, i, j in di['BRANCHES_AT_NODE'][node]:
                if i == node:
                    out[node, n, i, j] = -1
                elif j == node:
                    out[node, n, i, j] = 1
        return out
    di['Branch_dir_at_node'] = getBranchDirAtNode()

    return {'detData': di}
