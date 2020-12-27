# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:46:07 2019

@author: espenfb
"""

import pandas as pd
import numpy as np
import copy

bus = pd.read_csv('bus_data.csv', index_col = 0, skipinitialspace = True)

line = pd.read_csv('line_data_crez.csv', skipinitialspace = True)

R = 6378.1 # radius of earth in km

km2mile = 0.62137
R = R*km2mile

deg2rad = np.pi/180

def haversin(A):
    
    return np.power(np.sin(A*deg2rad/2),2)

for i in line.index:
    from_bus = line.iloc[i].From
    to_bus = line.iloc[i].To
    
    from_lon = bus.loc[from_bus].Lon
    from_lat = bus.loc[from_bus].Lat
    
    to_lon = bus.loc[to_bus].Lon
    to_lat = bus.loc[to_bus].Lat
    
    
    
    line.loc[i,'Distance'] = 2*R*np.arcsin(np.sqrt(haversin(to_lat - from_lat) + 
                                 np.cos(from_lat*deg2rad)*np.cos(to_lat*deg2rad)*haversin(to_lon - from_lon)))
    
line['Distance'] = np.round(line['Distance'],0)

# Add line 6 (1,9), line 26 (9,7) and line 23 (5,10) as possible investments
last_index = line.index[-1]
new_line = line.loc[6]
new_line.Type = 'New'
new_line.Distance = line.loc[6].Distance
line.loc[last_index+1] = new_line
new_line = line.loc[26]
new_line.Type = 'New'
new_line.Distance = line.loc[26].Distance
line.loc[last_index+2] = new_line
new_line = line.loc[23]
new_line.Type = 'New'
new_line.Distance = line.loc[23].Distance
line.loc[last_index+3] = new_line

line.loc[line.To < line.From,'from'] = line.To[line.To < line.From].to_list()
line.loc[line.To > line.From,'from'] = line.From[line.To > line.From].to_list()

line.loc[line.To > line.From,'to'] = line.To[line.To > line.From].to_list()
line.loc[line.To < line.From,'to'] = line.From[line.To < line.From].to_list()
    

inv_cost_mile_MW = 3000 # $/(mile*MW) CREZ at $2500/MW*mile
inv_cost_mile_MW_high = 4000 # $/(mile*MW) CREZ at $2500/MW*mile
terrain_factor = 1.0

mile2km = 1.6
kW2MW = 0.001

inv_cost_km = (inv_cost_mile_MW*terrain_factor)/(mile2km)
inv_cost_mile = (inv_cost_mile_MW*terrain_factor)
inv_cost_mile_high = (inv_cost_mile_MW_high*terrain_factor)

irr = 0.066
lifetime = 40.0 # years

epsilon = irr/(1-(1+irr)**(-lifetime))

ann_cost_km = inv_cost_km*epsilon
ann_cost_mile = inv_cost_mile*epsilon
ann_cost_mile_high = inv_cost_mile_high*epsilon

#line.loc[:,'Cost'] = line.loc[:,'Distance']*ann_cost_km 
line.loc[:,'Cost'] = (line.loc[:,'Distance']*ann_cost_mile)

new_line_high = copy.copy(line[line.Type == 'New'])
new_line_high.Type = 'New2'
new_line_high.Cost = (new_line_high.Distance*ann_cost_mile_high)
line = pd.concat([line, new_line_high])

exists = line.index[line.loc[:,'Type'] == 'Existing']


line.loc[exists,'Cost'] = 0.0

 

line.loc[:,'Cost'] = line.loc[:,'Cost'].round(0)

# Double cpacity
#new = line.index[line.loc[:,'Type'] == 'New']
# line.loc[new,'B'] = line.loc[new,'B']*2
# line.loc[new,'Cap'] = line.loc[new,'Cap']*2
# line.loc[new,'Cost'] = line.loc[new,'Cost']*2

line.to_csv('line_data_cost.csv', index = False)

pipes = copy.copy(line.loc[line[['from','to']].duplicated() == False])

a = 0.000466813	#[€/(m*(kg/h)*a)]
b = 37.3824	#€/(m*a)
#cap = 300000 # 300000 kg/h = 10 GW
cap = 1200000 # 300000 kg/h = 40 GW

pipes.Type = 'H2'
pipes.B = 0.0
pipes.Cap = cap
dist_m = pipes.Distance*1000/(mile2km)
pipes.Cost = dist_m*(a*cap+b)


# pipes_high = copy.copy(pipes)
# pipes_high.Cap = cap*2
# pipes_high.Cost = dist_m*(a*cap*2+b)
# pipes_high.Type = 'H2_2'

#lines_and_pipes = pd.concat([line,pipes,pipes_high])

lines_and_pipes = pd.concat([line,pipes])

lines_and_pipes.to_csv('lines_and_pipes.csv', index = False)


###### Merge duplicates ###


lap = lines_and_pipes

lap.loc[lap.To < lap.From,'from'] = lap.To[lap.To < lap.From].to_list()
lap.loc[lap.To > lap.From,'from'] = lap.From[lap.To > lap.From].to_list()

lap.loc[lap.To > lap.From,'to'] = lap.To[lap.To > lap.From].to_list()
lap.loc[lap.To < lap.From,'to'] = lap.From[lap.To < lap.From].to_list()

lap = lap.astype({"from": int, "to": int})

#lp_unique = []
#
lap.From = lap['from']
lap.To = lap['to']
lap = lap.drop(columns=['from','to'])

lap_aggr = lap.groupby(['From','To','Type']).agg({'Cap':'sum','B':'sum','Cost':'sum'}).reset_index()
#for i in lap.index:
#    br = lap.iloc[i]
#    fn = br.From
#    tn = br.To
#    tp = br.Type
#    count = ((lap.From == fn)&(lap.To == tn)&(lap.Type == tp))
#    count = count.sum()
#    lp_unique.append(count)
#lp_unique = np.array(lp_unique)
#lap_un = lap[(lap.duplicated() == False).to_list()]
#lap_un.loc[:,'Num'] = lp_unique[(lap.duplicated() == False).to_list()]
#
#lap_un.Cap = lap_un.Cap*lap_un.Num
#
#
#lap.B = lap.B**2/lapB*2

# SET MAX TRANS CAP TO 15 GW, IGNORE SUCEPTANCE 
new = lap_aggr.index[lap_aggr.loc[:,'Type'] == 'New']
lap_aggr.loc[new,'Cap'] = 5000
lap_aggr.loc[new,'Cost']  *= lap_aggr.loc[new,'Cap']

new2 = lap_aggr.index[lap_aggr.loc[:,'Type'] == 'New2']
lap_aggr.loc[new2,'Cap'] = 5000
lap_aggr.loc[new2,'Cost']  *= lap_aggr.loc[new2,'Cap']
lap_aggr.loc[new2,'Type'] = 'New' 

# pipes_high = lap_aggr.index[lap_aggr.loc[:,'Type'] == 'H2_2']
# lap_aggr.loc[pipes_high,'Type'] = 'H2' 

lap_aggr.to_csv('lines_and_pipes_aggr_2tier_p40.csv', index = False)
