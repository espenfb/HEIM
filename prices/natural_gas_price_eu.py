# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:00:35 2019

@author: espenfb
"""
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np

path = gpd.datasets.get_path('naturalearth_lowres')
df = gpd.read_file(path)

extent = [-25,45,35,72]

ax = gplt.polyplot(df[df.continent == 'Europe'],
                   projection=gcrs.AlbersEqualArea(central_longitude =23.145556 , central_latitude =53.135278  ),
                   facecolor='lightgray', edgecolor='black', linewidth=0, extent = extent)

prices = pd.read_excel('natural_gas_europe_extensive.xlsx',
                       skiprows = [0,1,2,3,4,5,6,7,8,9,10,11],
                       header = [0],
                       index_col = 0,
                       sheet_name = 'Data52')

df = df[df.name.isin(prices.index)]
df.sort_index(inplace = True)

indx = prices.index.isin(df.name)
prices = prices.loc[indx]
prices.sort_index(inplace = True)

#df.index = df.name
df['price'] = prices['2018S2'].tolist()

prices.transpose().plot()



gplt.choropleth(df, hue='price', scheme = 'map', cmap='Greens_r',
                projection=gcrs.AlbersEqualArea(central_longitude =23.145556 , central_latitude =53.135278),
                   linewidth=0.5, edgecolor='black', k=None, legend=True, ax = ax, extent = extent)

