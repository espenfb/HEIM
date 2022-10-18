# %%
import os
import sys
from pathlib import Path

filepath = os.path.abspath('../')
sys.path.append(str(Path(filepath) / "src"))

import detInvModel as dim
import pandas as pd
pd.set_option("plotting.backend","plotly")
import savedRes as sr

# %%
time_data = {
    'start_date': pd.Timestamp(year=2019, month=1, day=1),
    'end_date': pd.Timestamp(year=2020, month=1, day=1),
    'ref_date': pd.Timestamp(year=2019, month=1, day=1)}
dirs = {
    'data_dir': "data\\",
    'ctrl_data_file': 'ctrl_data.csv',
    'res_dir': 'results_test\\'}

# %%
obj = dim.deterministicModel(time_data, dirs)

# %%
res = sr.savedRes(dirs['res_dir'], data=obj.data)

# %%
res.getInvByType()

# %%
res.getLineInv()

# %%
nodes = res.getObjects("EL_NODES", default=res.INTERNAL_NODES)

# %%
nodes

# %%
hasattr(res, "MARKET_NODES")

# %%
res.NODES

# %%
res.getInv()

# %%
res.getInvByNode(nodes="EL_NODES")

# %% [markdown]
# # Plotting maps

# %%
res.plotMapInteractive()


# %%
if False:
    import geopandas as gpd
    pd.set_option("plotting.backend", "plotly")
    import cartopy.crs as ccrs
    file_name = "geo/ref-countries-2020-10m.shp/CNTR_RG_10M_2020_4326.shp/CNTR_RG_10M_2020_4326.shp"
    tx = gpd.read_file(file_name)
    no = tx[tx.CNTR_ID=="NO"]
    ax = plt.axes(projection=ccrs.PlateCarree())
    res.data.bus.plot(ax =ax)
    #no.plot(ax = ax, color='white', edgecolor='black', zorder=0)
    pad = 2
    ax.set_ylim(res.data.bus.Lat.min()-pad, res.data.bus.Lat.max()+pad)
    ax.set_xlim(res.data.bus.Lon.min()-pad, res.data.bus.Lon.max()+pad)

# %%
res.plotMapInv(node_color="black", line_nodes=["EL_NODES", "MARKET_NODES"])

# %%
map2 = res.plotMapInv(node_color="black", bus_objects="HYDRO_STORAGE", line_nodes=["EL_NODES", "MARKET_NODES"])

# %%
from IPython.display import display, HTML

htmlmap = HTML('<iframe srcdoc="{}" style="float:left; width: {}px; height: {}px; display:inline-block; width: 49%; margin: 0 auto; border: 2px solid black"></iframe>'
           '<iframe srcdoc="{}" style="float:right; width: {}px; height: {}px; display:inline-block; width: 49%; margin: 0 auto; border: 2px solid black"></iframe>'
           .format(map.get_root().render().replace('"', '&quot;'),500,500,
                   map2.get_root().render().replace('"', '&quot;'),500,500))
display(htmlmap)

# %%
file = open("sample.html", "w")
file.write(htmlmap._repr_html_())
file.close()


# %%
res.NODES

# %%
res.data.hydrogen_load

# %%
res.data.plant_char

# %%
res.data.plant_char

# %%
res.POWER_PLANT_TYPES

# %%
res.Fixed_energy_cost

# %%



