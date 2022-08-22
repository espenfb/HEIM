# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:22:31 2019

@author: espenfb
"""

import pandas as pd
import numpy as np

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import copy
from ast import literal_eval
import detModelRes as dmr

idx = pd.IndexSlice


class savedRes(object):

    def __init__(self, folder, data=None):

        self.lhv = 0.0333  # MWh/kg

        dmr.importRes(self, folder)

        for k, v in self.detData.items():
            if len(v) > 0:
                if list(v.keys())[0] is None:
                    setattr(self, k, v[None])
                else:
                    setattr(self, k, v)

        self.PLANT_AT_NODE = copy.copy(self.GEN_AT_NODE)
        for k, v in self.STORAGE_AT_NODE.items():
            if k in self.PLANT_AT_NODE.keys():
                self.PLANT_AT_NODE[k] += v
            else:
                self.PLANT_AT_NODE[k] = v

        self.h2power = pd.Series({i: self.lhv for i in
                                  self.HYDROGEN_PLANT_TYPES + ['Hydrogen']})

        self.NODE_TO_PLANT = {}
        for k, v in self.PLANT_AT_NODE.items():
            self.NODE_TO_PLANT.update({i: k for i in v})

        ch_name = {'Onshore Wind': 'Wind', 'Hydrogen': 'H$_2$'}

        self.PLANT_TO_TYPE = {}
        for k, v in self.TYPE_TO_PLANTS.items():

            if k in ch_name.keys():
                val = ch_name[k]
            else:
                val = k
            self.PLANT_TO_TYPE.update({i: val for i in v})

        self.data = data

        self.type_identifier = {v: k for k, v in self.type2prefix.items()}

        self.inv_res.sort_index(axis=1, inplace=True)
        self.addGeoBus()
        self.addGeoLine()

        self.s_ch = copy.copy(self.data.storage_char)
        self.s_ch.set_index('Type', inplace=True)

    # Convert ##
    def convH2power(self, df, axis=0):

        if axis == 0:
            indx = df.index.isin(self.h2power.index)
            df[indx] = df[indx].multiply(self.h2power, axis=0).dropna()
        elif axis == 1:
            indx = df.columns.isin(self.h2power.index)
            df[indx] = df[indx].multiply(self.h2power, axis=0).dropna()
        return df

    # GET ##
    def getObjects(self, objects, default=idx[:]):

        out = default
        if objects.__class__ == str:
            if hasattr(self, objects):
                out = getattr(self, objects)
        elif objects.__class__ == list:
            out = []
            for i in objects:
                if hasattr(self, i):
                    out += getattr(self, i)
        return out

    # Get - Investments
    def getInv(self, objects=None, cap_type='power', default_objects=idx[:]):

        objects = self.getObjects(objects, default=default_objects)

        if cap_type == 'power':
            retired_cap = ['retired_cap']
        else:
            retired_cap = []

        inv = self.inv_res.loc[idx[objects], idx[['Init_' + cap_type,
                               'new_' + cap_type] + retired_cap]].fillna(0)
        if cap_type == 'power':
            inv['retired_cap'] *= -1
        return inv

    def getInvByType(self, objects=None, cap_type='power', lower_lim=0.0,
                     default_objects=idx[:], conH2power=False):

        agg_by = {'Init_' + cap_type: 'sum', 'new_' + cap_type: 'sum'}
        if cap_type == 'power':
            agg_by.update({'retired_cap': 'sum'})

        inv = self.getInv(objects=objects, cap_type=cap_type,
                          default_objects=default_objects)
        inv_type = inv.rename(self.PLANT_TO_TYPE)
        inv_type = inv_type.reset_index().groupby('index')
        inv_type_sum = inv_type.agg(agg_by)
        inv_type_sum = inv_type_sum[inv_type_sum.max(axis=1) >= lower_lim]

        if conH2power:
            inv_type_sum = self.convH2power(inv_type_sum)

        return inv_type_sum

    def getInvByNode(self, nodes=None, cap_type='power', lower_lim=0.0):

        nodes = self.getObjects(nodes, default=self.NODES)

        inv_by_node = pd.DataFrame()
        for i in nodes:
            obj_filt = self.PLANT_AT_NODE[i]
            inv = self.getInvByType(objects=None, cap_type=cap_type,
                                    lower_lim=lower_lim,
                                    default_objects=obj_filt)
            inv_list = inv.index.get_level_values(0).tolist()
            inv.index = pd.MultiIndex.from_product([[i], inv_list])
            inv_by_node = pd.concat([inv_by_node, inv], axis=0)
        return inv_by_node

    # Get - Operations
    def getValue(self, val_type, objects=None, times=idx[:],
                 lower_lim=1.0, by_node=False):
        ''' Gets operational results and filter by time, object type
        and lower limit '''

        objects = self.getObjects(objects)

        vals = self.opr_res.loc[times, idx[objects, val_type]]
        vals = vals.loc[:, vals.max(axis=0) >= lower_lim]
        vals_col = vals.columns.get_level_values(0)
        if by_node:
            vals.columns = [int(i[-2:]) for i in vals_col]
        return vals

    def getValueByType(self, val_type, objects=None, times=idx[:],
                       lower_lim=1.0, sort_by_cf=True):
        
        vals = self.getValue(val_type, objects=objects,
                             times=times)
        vals = vals.rename(self.PLANT_TO_TYPE, axis=1).reindex(axis=1)
        vals = vals.groupby(axis=1, level=0).agg('sum')
        vals = vals.loc[:, vals.max(axis=0) >= lower_lim]

        if sort_by_cf: 
            cf = vals.mean()/vals.max()
            vals = vals[cf.sort_values(ascending=False).index]

        return vals

    def getValueByNode(self, val_type, nodes=None, objects=None,
                       times=idx[:], lower_lim=0.0):

        nodes = self.getObjects(nodes, default='NODES')
        objects = self.getObjects(objects, default=self.PLANTS)

        val_by_node = pd.DataFrame()
        for i in nodes:
            obj_filt = pd.Series(self.PLANT_AT_NODE[i])
            obj_filt = obj_filt[obj_filt.isin(objects)].tolist()
            vals = self.getValue(val_type, objects=obj_filt, times=times,
                                 lower_lim=lower_lim)
            vals_col_list = vals.columns.get_level_values(0).tolist()
            vals.columns = pd.MultiIndex.from_product([[i], vals_col_list])
            val_by_node = pd.concat([val_by_node, vals], axis=1)

        return val_by_node.rename(self.PLANT_TO_TYPE, axis=1)

    def getProd(self, val_type, coeff, objects=None):

        val = self.getValue(val_type, objects=objects)
        val = val.droplevel(1, axis=1)
        c = pd.Series(getattr(self, coeff))
        return val.multiply(c).dropna(axis=1).reindex()

    def getProdByType(self, val_type, coeff, objects=None):

        val = self.getValueByType(val_type, objects=objects)
        c = pd.Series(getattr(self, coeff))
        return val.multiply(c)

    def getWeightedValue(self, val_type, weight_type, val_objects=None,
                         weight_objects=None, val_coeff=None,
                         weight_coeff=None):

        if val_coeff is None:
            val = self.getValue(val_type, objects=val_objects)
            val = val.droplevel(1, axis=1)
        else:
            val = self.getProd(val_type, val_coeff, objects=val_objects)
        val.columns = [int(i[-2:]) for i in val.columns]

        if weight_coeff is None:
            weight = self.getValue(weight_type, objects=weight_objects)
            weight = weight.droplevel(1, axis=1)
        else:
            weight = self.getProd(weight_type, weight_coeff,
                                  objects=weight_objects)
        weight.columns = [int(i[-2:]) for i in weight.columns]

        weighted_val = (val*weight)/weight.sum()
        return weighted_val.sum()

    def getLineInv(self, nodes=None, aggr=True):

        nodes = self.getObjects(nodes, default=self.EL_NODES)

        con1 = self.inv_line.loc[idx[:], 'From'].isin(nodes)
        con2 = self.inv_line.loc[idx[:], 'To'].isin(nodes)
        index = con1 & con2
        lines = self.inv_line[index]
        if aggr:
            agg_by = {'Cap': 'sum', 'MaxCap': 'sum', 'geometry': 'first'}
            lines = lines.groupby(['From', 'To']).agg(agg_by)
        return lines

    def emissionByType(self):
        prod_sum = self.getValueByType('prod', objects='POWER_PLANTS').sum()
        emf = pd.Series(self.CO2_coef)

        return prod_sum.multiply(emf).sum()

    def emissionFromH2(self):
        prod_sum = self.getValueByType('prod', objects='H2_PLANTS').sum()
        emf = pd.Series(self.CO2_coef)

        return prod_sum.multiply(emf).sum()

    # Plots
    def plotValue(self, val_type, objects=None, times=idx[:],
                  lower_lim=0.0):
        ''' Plots operational results by value type'''

        vals = self.getValue(val_type, objects=objects, times=idx[:],
                             lower_lim=1.0)
        vals = vals.droplevel(1, axis=1)
        if not vals.empty:
            ax = vals.plot()

            if pd.get_option("plotting.backend") == "matplotlib":
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            ax.show()
        else:
            print("No values found!")

    def plotValueByNode(self, val_type, nodes=None, kind='bar',
                        objects=None, times=idx[:], lower_lim=0.0):

        val_by_node = self.getValueByNode(val_type, nodes=nodes,
                                          objects=objects, times=times,
                                          lower_lim=lower_lim)

        ax = val_by_node.unstack().T.plot(kind=kind)
        if pd.get_option("plotting.backend") == "matplotlib":
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    def plotValueByType(self, val_type, objects=None, times=idx[:],
                        lower_lim=0.0, sort_by_cf=True, kind='area',
                        **kwargs):

        val_by_type = self.getValueByType(val_type, objects=objects,
                                          times=times, lower_lim=lower_lim,
                                          sort_by_cf=sort_by_cf)

        ax = val_by_type.plot(kind=kind, **kwargs)
        if pd.get_option("plotting.backend") == "matplotlib":
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    def energySumByType(self, node='all'):

        if node == 'all':
            plants = self.PLANTS
        else:
            plants = self.GEN_AT_NODE[node]

        res = pd.DataFrame()
        for i in plants:
            for c in self.opr_res[i].columns:
                item_type = self.type_identifier[i[:-2]]
                if (item_type not in res.index) or (c not in res.columns):
                    res.loc[item_type, c] = self.opr_res[i, c].sum()
                else:
                    if np.isnan(res.loc[item_type, c]):
                        res.loc[item_type, c] = self.opr_res[i, c].sum()
                    else:
                        res.loc[item_type, c] += self.opr_res[i, c].sum()
        return res

    def energyByType(self, node='all'):

        if node == 'all':
            plants = self.POWER_PLANTS
            storage = self.EL_STORAGE
        else:
            plants = self.GEN_AT_NODE[node]
            storage = self.STORAGE_AT_NODE[node]

        res = pd.DataFrame()
        for i in plants:
            item_type = self.type_identifier[i[:-2]]
            if item_type not in res.columns:
                res[item_type] = self.opr_res[i, 'prod']
            else:
                res[item_type] += self.opr_res[i, 'prod']

        for i in storage:
            item_type = self.type_identifier[i[:-2]]
            if item_type not in res.columns:
                res[item_type] = self.opr_res[i, 'from_storage']
            else:
                res[item_type] += self.opr_res[i, 'from_storage']

        res.index = self.opr_res.index

        return res

    def plotenergySumByType(self, node='all'):
        energySumByType = self.energySumByType(node=node)
        energySumByType.plot(kind='bar')

    def plotInvByType(self, objects=None, plotType='bar',
                      subplots=False, lower_lim=0.0):

        objects = self.getObjects(objects, default=self.POWER_PLANTS)
        inv_by_type = self.getInvByType(objects=objects, lower_lim=lower_lim)

        plt.figure('Investments')
        ax = plt.gca()

        backend = pd.get_option("plotting.backend")
        if backend == "matplotlib":
            inv_by_type.plot(kind=plotType, subplots=subplots, ax=ax)
        elif backend == "plotly":
            inv_by_type.plot(kind=plotType)

    def plotInvByBus(self, nodes=None, lower_limit=0.0):

        load = pd.Series(self.detData['Load']).unstack(level=1).mean()

        df = self.getInvByNode(nodes=nodes)
        # remove entries with capacity under lower limit
        df = df[(df.sum(axis=1) < lower_limit) == False]
        df.Init_power = df.Init_power - df.retired_cap
        ncols = len(df.index.levels[0])
        # plotting
        wr = [len(df.loc[i].index) for i in df.index.levels[0]]
        fig, axes = plt.subplots(nrows=1,
                                 ncols=ncols,
                                 sharey=True,
                                 gridspec_kw={'width_ratios': wr})

        for i, row in enumerate(df.index.levels[0]):
            ax = axes[i]

            if str(row) in load.index:
                load_level = load.loc[str(row)]
            else:
                load_level = 0.0
            index_num = len(df.loc[(row,)])
            ax.plot([0, index_num], [load_level, load_level],
                    color='grey', zorder=0)

            df.loc[(row,)].plot(ax=ax, kind='bar', width=.8, stacked=True)

            ax.set_xlabel(row, weight='bold')
            ax.xaxis.set_label_coords(0.5, -0.2)
            ax.set_axisbelow(True)

            if i != (len(df.index.levels[0])-1):
                # ax.spines['left'].set_visible(False)
                ax.legend_.remove()
            else:
                ax.legend(['Average demand', 'Initial capacity',
                           'New capacity'])
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            if i == 0:
                ax.set_ylabel('Capacity [MW]', fontsize=12)

            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

        # make the ticklines invisible
        ax.tick_params(axis=u'x', which=u'x', length=0)
        plt.tight_layout()
        # remove spacing in between
        fig.subplots_adjust(wspace=0)  # space between plots

        plt.show()

    def getH2SourceBus(self):

        nodes = self.H2_NODES
        var = ['prod']

        h2_source = self.getValueByNode(var, nodes=nodes).sum().unstack()

        return h2_source

    def getH2ShareBus(self):

        h2_bus = self.getH2SourceBus().sum(axis=1)
        h2_total = self.getH2SourceBus().sum().sum()

        return h2_bus/h2_total

    def plotH2ByBus(self, plotType='bar', stacked=False):

        df = self.getH2SourceBus()

        df.plot(kind=plotType, stacked=stacked, alpha=0.8)
        plt.legend()

    def getLineRes(self):
        self.addGeoLine()
        agg_by = {'Cap': 'sum', 'geometry': 'first'}
        inv_line = self.inv_line.groupby(['From', 'To']).agg(agg_by)
        line_res_geo = gpd.GeoDataFrame(inv_line)
        return line_res_geo    

    def addGeoLine(self):

        self.inv_line.sort_index(inplace=True)
        line_data = self.inv_line.fillna(0) 

        if 'geometry' not in line_data.columns:
            line_gdf = gpd.GeoDataFrame(line_data)
            lines = []
            for l in range(len(line_gdf.index)):
                fn = int(line_gdf.iloc[l].From[-2:])
                tn = int(line_gdf.iloc[l].To[-2:])
                from_point = self.data.bus.loc[fn].Coordinates
                to_point = self.data.bus.loc[tn].Coordinates
                line = LineString(from_point.coords[:]+to_point.coords[:])
                lines.append(line)
            line_gdf['geometry'] = lines
            self.inv_line = line_gdf

    def addGeoBus(self):
        if 'Coordinates' not in self.data.bus.columns:
            buses = self.data.bus.set_index('Bus')
            buses['xy'] = list(zip(buses['Lon'], buses['Lat']))

            buses['Coordinates'] = buses['xy'].apply(Point)

            self.data.bus = gpd.GeoDataFrame(buses, geometry='Coordinates')

    def plotMap(self, linetype=None, linestyle='color', line_color='k',
                nodetype='energy', nodes=None, objects=None,
                line_lim=0.0, bus_lim=0.0,
                node_color='None', textcolor='royalblue', hatch='',
                edgecolors='k', textoffset=0.0,
                colormap='tab20c', lwidth=3.0,
                mkr_scaling=100, fillstyle='none', marker='o', alpha=1.0,
                print_cbar=True, fig=False, ax=False,
                max_node=False, max_line=False, plot_legend=True,
                ncol=1, legend_loc='best',
                labels=True, plot_shape=True, rel_lines=False,
                bbox_node=None, bbox_line=None, fontsize=15):

        nodes = self.getObjects(nodes, default=self.EL_NODES)

        if not ax and not fig:
            fig, axes = plt.subplots(nrows=1,
                                     ncols=1,
                                     sharey=True)
            fig.canvas.set_window_title('Map ' + ' ' + str(nodetype))
            ax = axes
        else:
            ax.set_aspect('equal')

        if objects:
            objects = self.getObjects(objects)
            if nodetype == 'power':
                end_label = ''
                ns = self.inv_res.new_power.loc[idx[objects]]
                if objects == self.HYDROGEN_STORAGE:
                    # unit = 'tonne/h'
                    # unit_weight = 1/1000 # tonne/ kg
                    unit_weight = 1/30000  # GWh/kg 
                    unit = 'GW'
                else:
                    unit = 'GW'
                    unit_weight = 1/1000  # GWh/ MWh
            elif nodetype == 'energy':
                end_label = ' Storage'
                ns = self.inv_res.new_energy.loc[idx[objects]]
                if objects == self.HYDROGEN_STORAGE:
                    # unit = 'tonne'
                    # unit_weight = 1/1000 # tonne/ kg
                    unit_weight = 1/30000  # GWh/kg
                    unit = 'GWh'
                else:
                    unit = 'GWh'
                    unit_weight = 1/1000  # GWh/ MWh
            ns.index = [int(i[-2:]) for i in ns.index]
            self.data.bus.loc[:, 'Size'] = ns

        if nodes:
            if nodes == self.H2_NODES:
                trans_name = 'Pipeline'
                # trans_unit = 'tonne/h'
                # unit_t_weight = 1/1000 # tonne/ kg
                unit_t_weight = 1/30000  # GWh/kg
                trans_unit = 'GW'
                t_unit_format = '%.1f'
            elif nodes == self.EL_NODES:
                trans_unit = 'GW'
                unit_t_weight = 1/1000  # GWh/ MWh
                trans_name = 'Transmission'
                t_unit_format = '%d'

        if plot_shape:
            file_name = "../geo/Texas_State_Boundary_Detailed/ \
                        Texas_State_Boundary_Detailed.shp"
            tx = gpd.read_file(file_name)
            tx.plot(ax=ax, color='white', edgecolor='black', zorder=0)

        if objects:
            size = self.data.bus['Size']
            if not max_node:
                max_node = size.max()
            rel_size = size/max_node
            self.data.bus.plot(ax=ax, color=node_color, hatch=hatch,
                               edgecolors=edgecolors, alpha=alpha,
                               linewidths=3.0, markersize=rel_size*mkr_scaling,
                               marker=marker, zorder=5)
            if plot_legend:
                ord_mag = len(str(int(max_node)))
                area_range = (np.array([0.3, 0.6, 1.0])*max_node)
                area_range = area_range.round((ord_mag-2)*-1)
                points = []
                label = []
                for area in area_range:
                    points.append(plt.scatter([], [], color=node_color,
                                              hatch=hatch,
                                              edgecolors=edgecolors,
                                              alpha=alpha, linewidths=3.0,
                                              s=(area*mkr_scaling)/max_node,
                                              marker=marker))
                    label.append(str(int(area*unit_weight)))
                plant_txt = self.PLANT_TO_TYPE[objects[0]]
                unit_txt = ' (' + unit + ')'
                title_txt = plant_txt + end_label + unit_txt
                fig.legend(points, label, scatterpoints=1, frameon=False,
                           labelspacing=1.1,
                           title=title_txt,
                           ncol=ncol, loc=legend_loc,
                           fontsize=fontsize,  title_fontsize=fontsize+1, 
                           columnspacing=0.9, bbox_to_anchor=bbox_node)

        if labels:
            for k, row in self.data.bus.iterrows():
                ax.annotate(s=k, xy=(row['xy'][0] + textoffset,
                                     row['xy'][1] + textoffset),
                            color=textcolor,
                            fontsize=fontsize-1, weight='heavy')

        if linetype is None:
            print_cbar = False
        elif linetype == 'base':
            print_cbar = False
            self.addGeoLine()
            line_data = self.inv_line[self.inv_line.From.isin(nodes)]
            line_data['marker'] = 1
            line_data.plot(ax=ax, column='marker', cmap=colormap,
                           linewidth=lwidth)
        else:
            self.addGeoLine()
            line_data = self.inv_line[self.inv_line.From.isin(nodes)]
            group_by = ['From', 'To']
            agg_by = {linetype: 'sum', 'MaxCap': 'sum', 'geometry': 'first'}
            line_data = line_data.groupby(group_by).agg(agg_by)
            line_data = gpd.GeoDataFrame(line_data)
            line_data = line_data[line_data[linetype] > line_lim]
            if rel_lines: 
                line_data.Cap = line_data.Cap/line_data.MaxCap
            else:
                line_data[linetype] = line_data[linetype]
            vmin = line_data[linetype].min()
            vmax = line_data[linetype].max()
            if linestyle == 'color':
                line_data.plot(ax=ax, column=linetype, cmap=colormap,
                               vmin=vmin, vmax=vmax, linewidth=lwidth)
            elif linestyle == 'width':
                print_cbar = False
                if not max_line:
                    max_line = line_data.Cap.max()
                rel_line = line_data.Cap/max_line
                line_data.plot(ax=ax, color=line_color, linewidth=rel_line*20,
                               zorder=1)

                if plot_legend:
                    ord_mag = len(str(int(max_line)))
                    a_range = np.array([0.3, 0.6, 1.0])*max_line
                    round_by = (ord_mag - 2)*-1
                    area_range = np.nan_to_num(a_range).round(round_by)
                    points = []
                    label = []
                    for area in area_range:
                        print(area)
                        plt.plot([], [], c=line_color,
                                 linewidth=(area*20)/max_line,
                                 label=t_unit_format % (area*unit_t_weight))
                    fig.legend(frameon=False, labelspacing=1.1,
                               title=trans_name + ' (' + trans_unit + ')',
                               ncol=ncol, bbox_to_anchor=bbox_line,
                               loc=legend_loc, fontsize=fontsize,
                               title_fontsize=fontsize+1, columnspacing=0.9)

        ax.axis('off')

        if print_cbar:
            sm = plt.cm.ScalarMappable(cmap=colormap,
                                       norm=plt.Normalize(vmin=vmin,
                                                          vmax=vmax))
            sm._A = []

            fig.subplots_adjust(right=0.9)
            cax = fig.add_axes([0.93, 0.1, 0.01, 0.8])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(trans_name + ' capacity [' + trans_unit + ']')
        plt.tight_layout(pad=0.0)
        plt.show()
