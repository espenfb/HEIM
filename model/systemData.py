# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:42:36 2018

@author: espenfb
"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta 
import data_series_tools as dst
import production_records as pr
import parseFunctions as pf

import matplotlib.pyplot as plt

GW2MW = 1000
KW2MW = 0.001

class systemData(object):
    
    def __init__(self, dirs):
        
        delim = ','
        
        for k in dirs.keys():
            setattr(self, k, dirs[k])
        
        self.ctrl_data = pd.read_csv(self.data_dir + self.ctrl_data_file,
                                     delimiter = delim, skipinitialspace=True)
        ctrl_data = self.ctrl_data
        
        
        self.hydro_power = pd.read_csv(self.data_dir + 
                                 ctrl_data.loc[ctrl_data.type =='hydro','path'].values[0],
                                 skipinitialspace=True)
        self.hydro_power['id'] = [ 'H'+str(i) for i in self.hydro_power.bus_indx.tolist()]
        
        
        self.wind_power = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'wind','path'].values[0],
                                     skipinitialspace=True)
        self.wind_power['id'] = ['W'+str(i) for i in self.wind_power.bus_indx.tolist()]

        self.consumer = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'load','path'].values[0],
                                         skipinitialspace=True)
        self.consumer['id'] = ['C'+str(i) for i in self.consumer.bus_indx]

        self.hydrogen = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'hydrogen','path'].values[0],
                                         skipinitialspace=True)
        self.hydrogen['id'] = ['E'+str(i) for i in self.hydrogen.bus_indx]

        self.bus = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'bus','path'].values[0],
                                    skipinitialspace=True)

        self.line = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'line','path'].values[0],
                                     skipinitialspace=True)

        self.series = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'series','path'].values[0],
                                       skipinitialspace=True)

        series_data = self.series
        self.prices = pd.read_excel(self.data_dir + series_data.loc[series_data.type == 'price','path'].values[0],
                    header=2, sheet_name=series_data.loc[series_data.type == 'price','tab'].values[0],
                    index_col = [0], parse_dates = [0], tz_localize = 'utc')
        
        # set index to timestamp with hourly resolution
        self.prices.index = pd.DatetimeIndex([self.prices.index[i]
        + relativedelta(hours = int(self.prices['Hours'][i][:2])) for i in range(len(self.prices.index))])
        
        # Remove unwanted data
        self.prices.drop('Hours', axis = 1, inplace = True)
        self.prices.dropna(how = 'all', inplace = True)
    
        #localize timezone of datetime index
        self.prices.index = self.prices.index.tz_localize('Europe/Oslo', ambiguous = 'infer')
        # change timezone to utc
        self.prices.index = self.prices.index.tz_convert('utc')

        self.param = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'parameters','path'].values[0], skipinitialspace=True)
        
        # Scenario generator parameters
        self.scen_param = pd.read_csv(self.data_dir + ctrl_data.loc[ctrl_data.type == 'scen_param','path'].values[0], skipinitialspace=True)
        
        
    def importTimeSeries(self, start_date, end_date, loadScenGen = True,
                         sampleMethod = 'sample', reduce = False,
                         prod_type = 'real'):
        ''' Imports and stores full(yearly) time series to be used in the two-stage
        scenario fan. sampleMethod is 'sample', 'to_file' or 'from_file' '''
        
        self.start_date  = start_date
        self.end_date = end_date
        self.prod_type = prod_type
        
        max_length = 8760
    

        # REALIZED WIND POWER
        series_data = self.series
        data_dir = self.data_dir
        wind_series = pr.getProdByDate(load_dir = self.data_dir + series_data.loc[series_data.type == 'wind_real','path'].values[0])
        wind_farms = self.wind_power.name[self.wind_power.name.duplicated() == False].values
        
        self.wind_power_production = pd.DataFrame()
        for i in wind_farms:
            self.wind_power_production[i] = wind_series[i]      
        
        
        # PRICE SERIES
        bus_data = self.bus
        prices = self.prices
        self.market_price = pd.DataFrame()
        market_bus = self.bus[self.bus.type == 'M'].index.tolist()
        for i in market_bus:
            area = bus_data.loc[i,'market_area']
            self.market_price[i] = prices[area].loc[
                        (prices.index >= start_date)&(prices.index < end_date + relativedelta(hours = self.horizon))]
            
            
        
        # INFLOW SERIES
        hydro_data= self.hydro_power
        
        inflow = pd.DataFrame()
        inflow_ureg = pd.DataFrame()
        inflow_series = {}
        for i in hydro_data.id:
            hydro_indx = hydro_data.index[hydro_data.id == i].values[0]
            series_nr = hydro_data.at[hydro_indx,'seriesNr']
            if series_nr not in inflow_series.keys():
                inflow_series_temp = pd.read_excel(data_dir + series_data.loc[series_nr,'path'],
                        header=0, sheet_name=series_data.loc[series_nr,'tab'])
                for j in inflow_series_temp.keys():
                    inflow_series[j] = dst.splinalInterpolation(inflow_series_temp[j].tolist(), 168)
                    
            inflow[i] = pd.Series((inflow_series[series_nr]*hydro_data.at[hydro_indx,'inflow_tot_year']*GW2MW)[:max_length])
            inflow_ureg[i] = pd.Series((inflow_series[series_nr]*hydro_data.at[hydro_indx,'inflow_ureg_year']*GW2MW)[:max_length])
            
        self.inflow = inflow
        self.inflow_ureg = inflow_ureg
        
        consumer_data = self.consumer
        
        #LOAD SERIES
        consumer_load = pd.DataFrame()
        s_indx = series_data.index[series_data.type == 'load'].values[0]
        path = data_dir + series_data.at[s_indx ,'path']
        tab = series_data.at[s_indx ,'tab']
        temp_series = pd.read_excel(path, index_col = 0, header=0,
                            parse_dates = True, sheet_name=tab)

        # set index to timestamp with hourly resolution
        temp_series.index = pd.DatetimeIndex([temp_series.index[i]
                    + relativedelta(hours = int(temp_series['Hours'][i][:2]))
                    for i in range(len(temp_series.index))])
        
        # Remove unwanted data
        temp_series.drop('Hours', axis = 1, inplace = True)
        temp_series.dropna(how = 'all', inplace = True)
    
        #localize timezone of datetime index
        temp_series.index = temp_series.index.tz_localize('Europe/Oslo', ambiguous = 'infer')
        # change timezone to utc
        temp_series.index = temp_series.index.tz_convert('utc')
        
        temp_series = temp_series.loc[(temp_series.index >= start_date)&(temp_series.index < end_date + relativedelta(hours = self.horizon))]
        for i in consumer_data.id:
            c_indx = consumer_data.index[consumer_data.id == i].values[0]
            #year = int(consumer_data.get_value(c_indx, 'year'))
            normal_load = float(consumer_data.at[c_indx, 'normal'])
            const_load = float(consumer_data.at[c_indx, 'constant'])
            consumer_series = temp_series.iloc[:,0]
            consumer_load[i] = GW2MW*(consumer_series*normal_load + const_load/8760)
            
        self.load = consumer_load
        
        
        # HYDRO POWER RESERVOIR CURVES
        hydro_power_res_file = data_dir + series_data.loc[series_data.type == 'hydro_res','path'].values[0]
        hydro_power_res = pd.read_csv(hydro_power_res_file, index_col = 0,
                                      parse_dates = [0])
        hydro_power_res.index = hydro_power_res.index.tz_localize('utc')

        reservoir = pd.DataFrame()
        #water_value = pd.DataFrame()
        for i in hydro_power_res.keys():
            reservoir[i] =  pd.Series(hydro_power_res[i])
            #water_value = pd.Series(hydro_power_res[i]['water_value'])
        self.reservoir_curve = reservoir
        #self.water_value = water_value
        
        # HYDROGEN DEMAND
        self.hydrogen_demand_series = pd.DataFrame()
        for i in self.hydrogen.id:
            indx = self.hydrogen.index[self.hydrogen.id == i].values[0]
            series_nr = int(self.hydrogen.at[indx, 'seriesNr'])
            series = pd.read_excel(data_dir + series_data.loc[series_data.seriesNr == series_nr,'path'].values[0],
                        header=0, sheet_name=series_data.loc[series_data.seriesNr == series_nr,'tab'].values[0])
            series = dst.appendSeries(dst.scaleSeries(series[series.keys()[0]],168),24,1)
            self.hydrogen_demand = float(series[0])
            self.hydrogen_demand_series[i] = pd.Series(series)
            
        
        
    def plotSeries(self, series_type, x_range = 'all', save = False, import_dir = None):
        
        res = getattr(self, series_type)
        
        for k in res.columns:
            plt.figure(str(k))
            ax = plt.gca()
            if x_range == 'all':
                line, = ax.plot(res.loc[:,k])
                line.set_label(k + '_' + series_type)
            else:
                line, = ax.plot(res.loc[pd.date_range(start = x_range[0].tz_convert(None),
                                                      end = x_range[-1].tz_convert(None),
                                                      freq = 'H'),k].tolist())
                line.set_label(k + '_' + series_type)
            plt.legend()
            
            if save:
                plt.savefig(import_dir + str(k) +'.pdf')
    
