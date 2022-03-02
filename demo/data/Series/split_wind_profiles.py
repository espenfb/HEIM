# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:22:05 2020

@author: espenfb
"""

import pandas as pd
idx = pd.IndexSlice

wp = pd.read_csv('wind_profiles_adj.csv',
				  index_col = 0,
				  header = [0,1])


wp_inst = wp.loc[idx[:],idx[:,'Inst_cap']]
wp_inst = wp_inst.droplevel(1,axis = 1)

wp_pot = wp.loc[idx[:],idx[:,'Pot_cap']]
wp_pot = wp_pot.droplevel(1,axis = 1)



wp_inst.to_csv('inst_wind_profile.csv')
wp_pot.to_csv('pot_wind_profile.csv')