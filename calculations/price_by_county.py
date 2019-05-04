# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:59:48 2019

@author: espenfb
"""

import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import matplotlib.pyplot as plt
import cartopy
from shapely.geometry import Point

ax_ng = plt.axes(projection=cartopy.crs.PlateCarree())

ng = gpd.read_file('..\\geo\\H_NGasCostInd_2015\\H_NGasCostInd_2015.shp')

ng = ng[ng.state_name == 'TX']
ng.index = ng.name
ng.sort_index(inplace = True)

gplt.choropleth(ng, hue='ind_dmcf', scheme = 'map', cmap='Greens',
                projection=gcrs.AlbersEqualArea(),
                   linewidth=0.5, edgecolor='black', k=None, legend=True,
                   ax = ax_ng)
ax_ng.set_title('Natural gas prices [$/MCF]')

##############################################################################

mmBtu2MCF = 1.037
mmBtu_per_kg = 0.1558

cost_ccs = 0.0 # $/kg H2

cost_param_ng = pd.read_csv('..\\hydrogen_production\\NATURAL_GAS\\cost_param_central_cs.csv',
                         skipinitialspace = True, index_col = 0)

cost_param_ng.drop(labels = 'Feedstock Costs', inplace = True)

ng['central_ng_cs'] = ng['ind_dmcf']*mmBtu_per_kg*mmBtu2MCF + cost_param_ng.sum()[0] + cost_ccs

plt.figure()
ax_ng_cs = plt.axes(projection=cartopy.crs.PlateCarree())

gplt.choropleth(ng, hue='central_ng_cs', scheme = 'map', cmap='Greens',
                projection=gcrs.AlbersEqualArea(),
                   linewidth=0.5, edgecolor='black', k=None, legend=True,
                   ax = ax_ng_cs)
ax_ng_cs.set_title('Hydrogen Cost NG [$/kg]')

##############################################################################


el = gpd.read_file('..\\geo\\H_ElectricCostInd_2016\\H_ElectricCostInd_2016.shp') 

plt.figure()
ax_el = plt.axes(projection=cartopy.crs.PlateCarree())

el = el[el.state_name == 'TX']
el.index = el.name
el.sort_index(inplace = True)


gplt.choropleth(el, hue='ind_dmwh', scheme = 'map', cmap='Greens',
                projection=gcrs.AlbersEqualArea(),
                   linewidth=0.5, edgecolor='black', k=None, legend=True,
                   ax = ax_el)

ax_el.set_title('Electricity prices [$/MWh]')

##############################################################################

el_consumption = 54.3 #kWh/kg 
kWh2MWh = 0.001

cost_param = pd.read_csv('..\\hydrogen_production\\\PEM\\central_pem_cost.csv',
                         skipinitialspace = True, index_col = 0)

cost_param.drop(labels = 'Feedstock Costs', inplace = True)

el['central_elec'] = el['ind_dmwh']*el_consumption*kWh2MWh + cost_param.sum()[0]

plt.figure()
ax_c = plt.axes(projection=cartopy.crs.PlateCarree())
gplt.choropleth(el, hue='central_elec', scheme = 'map', cmap='Greens',
                projection=gcrs.AlbersEqualArea(),
                   linewidth=0.5, edgecolor='black', k=None, legend=True,
                   ax = ax_c)
ax_c.set_title('Hydrogen Cost ELEC current [$/kg]')




sum(ng.central_ng_cs > el.central_elec)


##############################################################################


cost_param_future = pd.read_csv('..\\hydrogen_production\\\PEM\\central_pem_cost_future.csv',
                         skipinitialspace = True, index_col = 0)

cost_param_future .drop(labels = 'Feedstock Costs', inplace = True)

el['central_elec_future'] = el['ind_dmwh']*el_consumption*kWh2MWh + cost_param_future.sum()[0]

plt.figure()
ax_c_f = plt.axes(projection=cartopy.crs.PlateCarree())
gplt.choropleth(el, hue='central_elec', scheme = 'map', cmap='Greens',
                projection=gcrs.AlbersEqualArea(),
                   linewidth=0.5, edgecolor='black', k=None, legend=True,
                   ax = ax_c_f)
ax_c_f.set_title('Hydrogen Cost ELEC future [$/kg]')




sum(ng.central_ng_cs > el.central_elec_future)



tx = gpd.read_file('..\\geo\\Texas_County_Boundaries_Detailed\\Texas_County_Boundaries_Detailed.shp')

tx.index = tx.CNTY_NM
tx = tx[(tx.index.isin(el.index.tolist() + ng.index.tolist()))]

tx['cheapest'] = 'Natural Gas'
tx.loc[ng.central_ng_cs > el.central_elec_future,'cheapest'] = 'Electrolysis'

tx.plot(column = 'cheapest', cmap = 'seismic', legend = True)


##############################################################################


cap = pd.read_excel('..\\production_capacity\\CapacityDemandandReservesReport-Dec2018.xlsx',
                    sheet_name = 'WinterCapacities', skiprows = [0,2], index_col = 0)

cap.dropna(subset = ['COUNTY'], inplace = True)
cap.dropna(subset = ['FUEL'], inplace = True)
cap.index = [int(i) for i in cap.index]
cap.COUNTY = cap.COUNTY.str.title()
cap.COUNTY.replace('Ft. Bend', 'Fort Bend', inplace = True)
cap.COUNTY.replace('Mclennan', 'McLennan', inplace = True)
cap.COUNTY.replace('Mcculloch', 'McCulloch', inplace = True)

sum_df = cap.groupby(['COUNTY','FUEL']).agg({'2028/2029': 'sum'})
wind = sum_df.xs('WIND', level=1, drop_level=True)

el['central_elec_future_wind'] = (el['ind_dmwh'] - 0.1*wind['2028/2029'])*el_consumption*kWh2MWh + cost_param_future.sum()[0]

tx['cheapest_wind'] = 'Natural Gas'
tx.loc[ng.central_ng_cs > el.central_elec_future_wind,'cheapest_wind'] = 'Electrolysis'

tx.plot(column = 'cheapest_wind', cmap = 'seismic', legend = True)



##############################################################################

texas = gpd.read_file('..\\geo\\Texas_County_Boundaries_line\\Texas_County_Boundaries_line.shp')

centroid = pd.read_csv('..\\geo\\Texas_Counties_Centroid_Map.csv')
centroid['Coordinates'] = list(zip(centroid['X (Lat)'], centroid['Y (Long)']))
centroid['Coordinates'] = centroid['Coordinates'].apply(Point)
centroid_gdf = gpd.GeoDataFrame(centroid, geometry = 'Coordinates')


fig = plt.figure('Wind and Solar')
ax = plt.gca()
texas.plot(ax = ax)
wind = pd.read_csv('..\\energy_profiles\\wind_cite_id.csv', skiprows = [0], skipinitialspace = True)
wind['Coordinates'] = list(zip(wind['LONGITUDE'], wind['LATITUDE']))
wind['Coordinates'] = wind['Coordinates'].apply(Point)
wind_gdf = gpd.GeoDataFrame(wind, geometry = 'Coordinates')
wind_gdf.plot(ax = ax, color = 'g')


county_coord = centroid.loc[:,['Coordinates','CNTY_NM']]
county_coord.rename(columns={'CNTY_NM':'County'}, inplace = True)

solar = pd.read_csv('..\\energy_profiles\\solar_cite_id.csv')

solar = pd.merge(solar, county_coord, on = ['County'], how = 'inner' )
solar_gdf = gpd.GeoDataFrame(solar, geometry = 'Coordinates')
solar_gdf.plot(ax = ax, color = 'y')
ax.legend(['county boundaries','buses', 'wind','solar'])
ax.set_title('Wind and Solar')


