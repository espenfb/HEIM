# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:24:33 2017

@author: Espen Flo Bødal
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import os
import copy
import parseFunctions as pf
import pandas as pd

_QUOTINGTYPE=csv.QUOTE_MINIMAL




def comma2dot(cell):
	cell = str(cell).replace(',','.')
	return cell



#def loadWindProductionFile(filename):
#
#	readHeaders = True
#	data = {}
#
#	with open(filename, 'r', newline='', encoding="latin1") as csvfile:
#			reader = csv.reader(csvfile, delimiter=';',quotechar='"', quoting=_QUOTINGTYPE)
#
#			for row in reader:
#				if readHeaders:
#					keys = row[:]
#					data_size = len(row)
#					readHeaders = False
#					for i in range(data_size):
#						data[keys[i]] = np.array([])
#				else:
#					data[keys[0]] = np.append(data[keys[0]],pf.parseDate(row[0]))
#					for i in range(1,data_size-1):
#						data[keys[i]] = np.append(data[keys[i]],pf.parseNum(comma2dot(row[i]), default = 1))
#	return data

def loadWindProductionFile(file, drop_modelled = True):
    #wind = pd.read_csv(file, delimiter = ';', decimal = ',', index_col = 'Date/Time', parse_dates = True)
    excel_file = pd.ExcelFile(file)
    wind = pd.read_excel(excel_file, index_col = 'Date/Time', parse_dates = True)
    
    wind.index.rename('Time', inplace = True)
    
    new_col = {}
    for col in  wind.columns:
        temp_col = col
        temp_col = temp_col.replace('aa','å')
        temp_col = temp_col.replace('oe','ø')
        temp_col = temp_col.replace('ae','æ')
        temp_col = temp_col.replace('(MWh/h)','')
        temp_col = temp_col.replace('g_j','gj')
        
        new_col[col] = temp_col
     
    wind = wind.rename(columns = new_col)
        
#    wind.index = pd.DatetimeIndex(wind.Time)
#    wind = wind.drop('Time', axis = 1)
    
    if drop_modelled:
        drop_index = ['modellert' in col for col in wind.columns]
        wind.drop(wind.columns[drop_index], axis = 1, inplace = True)
        wind.dropna(how = 'all', inplace = True)
    
    return wind


def determine_erronious_val_limit(series):
	error_threshold = np.inf
	sorted_series = copy.copy(series)
	sorted_series.sort()
	for i in range(len(sorted_series)-1):
		if sorted_series[i+1]-sorted_series[i] > 1.0:
			error_data_indx = i
			error_threshold = sorted_series[error_data_indx]
			break
	return error_threshold

def cleanProductionSeries(series):
	series_start_indx = 0
	window_width =1000
	for i in range(len(series)):
		if i >= window_width and i <= (len(series)-window_width):
			prev_max = np.max(series[i-window_width:i])
			if prev_max <= 1.0:
				series_start_indx = i
	#print(series_start_indx)
	temp_clean_series = copy.copy(series[series_start_indx:len(series)])

	max_value = determine_erronious_val_limit(temp_clean_series)
	found_real_start = False
	changed_values_indx = []
	real_start = 0

	for i in range(len(temp_clean_series)):
		if temp_clean_series[i] > 0.6*max_value and found_real_start == False:
			real_start = i
			#print(real_start)
			found_real_start = True
		if temp_clean_series[i] > max_value:
			temp_clean_series[i] = np.mean(temp_clean_series[i-4:i])
			changed_values_indx.append(i)
	#print(changed_values_indx)
	start_indx = series_start_indx + real_start

	return (temp_clean_series[real_start:], start_indx)

def findMaxVal(df):
    out = {}
    for col in df.columns:
        temp = df[col].sort_values().dropna().tolist()[::-1]
        max_val = max(temp)
        skip_cntr = 0
        for i in temp:
            if max_val - i > max_val*0.05:
                max_val = i
                skip_cntr = 0
            skip_cntr += 1
            if skip_cntr > 100:
                out[col] = max_val
                break
    return out

def findSeriesStart(df, max_val, start_lim = 0.9):
    out = {}
    for col in df.columns:
        out[col] = df[col][df[col] > start_lim*max_val[col]].index[0]
    return out

def setTimezoneUTC(df):
    years = df.index.year.unique()
    
    out = pd.DataFrame(columns = df.columns, index=pd.DatetimeIndex([]))

    for year in years:
        print('Year: ', year)
        
        full_year = pd.date_range('01-01-' + str(year), '31-12-' + str(year),
                                  freq = 'H')
        data_year = copy.copy(df[df.index.year == year])
        
        missing_dates =  full_year[full_year.isin(data_year.index) == False]
        print('...missing dates: ', missing_dates.tolist())
        
        if sum(missing_dates.month == 3) == 1:
            data_year.index = data_year.index.tz_localize('Europe/Helsinki', ambiguous = 'NaT')
            data_year.index = pd.DatetimeIndex(data_year.index.tz_convert(datetime.timezone.utc))
            
            merged_dates = full_year[full_year.isin(data_year[data_year.index.year == year].index) == False]
            merged_dates = merged_dates.tz_localize('utc')
#            merged_dates_eu = merged_dates_utc.tz_convert('Europe/Helsinki')
            print('...merged dates: ', merged_dates.tolist())
            for d in merged_dates:
                b = d.astimezone('Europe/Helsinki')
                date = pd.datetime(b.year,b.month,b.day,b.hour)
                data_year.loc[d] = df.loc[date]*0.5
        else:
            new_index = []
            for d in data_year.index:
                new_index.append(d - relativedelta(hours = 2) )
            data_year.index = pd.DatetimeIndex(new_index)
            data_year.index = data_year.index.tz_localize('utc')   
            
        if len(out) == 0:
            out = data_year
        else:
            out = pd.concat([out,data_year])
            
    out.sort_index(inplace = True)
    out.drop('NaT', inplace = True)
    return out
            
            
def cleanDataFrame(df, relative = False):
    
    df = setTimezoneUTC(df)

    cap = findMaxVal(df)
    
    start_index = findSeriesStart(df, cap)
    
    for col in df.columns:
        viol_indx = df.index[df[col] > cap[col]]
        for i in viol_indx:
            df.loc[i, col] = min(df.loc[i-relativedelta(hours=1), col],cap[col])
        
        df.loc[df.index < start_index[col], col] = np.NaN
        if relative:
            df[col] = df[col]/cap[col]
        
    return df

#def cleanDataFrame(df):
#    
#    for key in df.keys():
#        if key == 'Time':
#            continue
#        (temp_series, start_indx) = cleanProductionSeries(np.array(df[key].tolist()))
#        if start_indx > 0:
#            df.loc[:start_indx,key] = [np.nan]*(start_indx+1)
#        print(key)
#        print('Start: %s' % df.index[start_indx])
#        print('End: %s' % df.index[start_indx+len(temp_series)-1])
#        df.loc[start_indx:start_indx+len(temp_series),key] = temp_series
#        if key== 'Raggovidda':
#            plt.figure()
#            plt.plot(temp_series)
#            plt.figure()
#            plt.plot(df[key])
#            print(df[key])
#    return df

def monthRange(start_date, end_date):
	if start_date.year == end_date.year:
		if end_date.month == start_date.month:
			monthrange = range(1)
		else:
			monthrange = range(int(end_date.month-start_date.month))
	elif start_date.year < end_date.year:
		monthrange = range(int(12-start_date.month)+int(end_date.month)+int((end_date.year-start_date.year-1)*12))
	return monthrange


def makeProductionSeriesFile(series_name, data_series, time_series, file_path):

	monthrange = monthRange(time_series[0],time_series[-1]+relativedelta(months = 1))

	for i in monthrange:
		if i == 0:
			start_date = time_series[0]
			end_date = datetime.datetime(time_series[0].year,time_series[0].month,1) + relativedelta(months = 1)
		else:
			start_date = datetime.datetime(time_series[0].year,time_series[0].month,1) + relativedelta(months = i)
			end_date = start_date + relativedelta(months = 1)

		directory = file_path + start_date.strftime('%Y') + '/' + start_date.strftime('%m') + '/'

		if not os.path.exists(directory):
			os.makedirs(directory)

		if series_name.find(' (MWh/h)') is not -1:
			series_name = series_name[0:series_name.find(' (MWh/h)')] + '_MW'
		elif series_name.find('(MWh/h)') is not -1:
			series_name = series_name[0:series_name.find('(MWh/h)')] + '_MW'

		filename = directory + series_name + '.csv'

		write_indx = (time_series >= start_date)*(time_series < end_date)

		time_series_str = []
		for time in time_series[write_indx]:
			time_series_str.append(str(time))

		with open(filename, 'w', newline='', encoding="latin1") as csvfile:
			datawriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=_QUOTINGTYPE)

			datawriter.writerow(['time'] + time_series_str)
			datawriter.writerow(['production'] + data_series[write_indx].tolist())
            
            
def makeYearlyProdFiles(data_file, save_dir = 'Production/', relative = True):
    
    data = loadWindProductionFile(data_file)
    
    data = cleanDataFrame(data, relative = relative)
    
    rel_str = ''
    if relative:
        rel_str = 'rel_'
    
    years = data.index.year.unique()
    
    for year in years:
        print(year)
        data_year = data[data.index.year==year]
        data_year.to_csv(save_dir + 'Wind_power_prod_' + rel_str + str(year) + '.csv')
        
        
def loadYearlyProd(year, load_dir = 'Production/', relative = False):
    
    rel_str = ''
    if relative:
        rel_str = 'rel_'
    
    data = pd.read_csv(load_dir + 'Wind_power_prod_' + rel_str + str(year) + '.csv',
                       encoding = 'latin-1', index_col = 0, parse_dates = True)
    
    return data

def getProdByDate(start_date, end_date, locations = 'all', load_dir = 'Production/', relative = True):
    
    years = np.arange(start_date.year, end_date.year+1)
    
    for i in years:
        if i == years[0]:
            data = loadYearlyProd(i, load_dir = load_dir, relative = relative)
        else:
            new_data = loadYearlyProd(i, load_dir = load_dir, relative = relative)
            data = pd.concat([data, new_data])
            
    if locations != 'all':
        data = data.iloc[:,data.columns.get_level_values(0).isin(locations)]
    # Set timezone
    #data.dropna(inplace = True)
    data.index = data.index.tz_localize('utc')
#    data.index = pd.DatetimeIndex(data.index.tz_convert(datetime.timezone.utc), freq = 'H')
    
    include = (data.index >= start_date)&(data.index <= end_date)    
    
    df = data.loc[include]
    
    return df

def saveProductionData(data, file_path = 'Production/'):

	for key in data.keys():
		if key != 'Date/Time':
			print(key)
			(clean_series, start_indx) = cleanProductionSeries(data[key])
			if len(clean_series) > 0:
				makeProductionSeriesFile(key, clean_series, data['Date/Time'][start_indx:], file_path)

def loadProductionFile(filename):

	time = np.array([])
	production = np.array([])

	with open(filename, 'r', newline='', encoding="latin1") as csvfile:
		reader = csv.reader(csvfile, delimiter=',',quotechar='"', quoting=_QUOTINGTYPE)
		for row in reader:
			if row[0] == 'time':
				time = np.append(time, np.array(pf.parseDatetimeList(row[1:])))
			elif row[0] == 'production':
				production = np.append(production, np.array(pf.parseNumList(row[1:])))

	return {'time':time,'production': production}

def loadProductionSeries(series_name, start_date,end_date, file_path = 'Production/'):
	time = np.array([])
	production = np.array([])

	monthrange = monthRange(start_date, end_date)
	#print(monthrange)

	for n in monthrange:
		current_date = start_date + relativedelta(months =n)

		filename = file_path + current_date.strftime('%Y') + '/' + current_date.strftime('%m') + '/' + series_name + '.csv'

		with open(filename, 'r', newline='', encoding="latin1") as csvfile:
			reader = csv.reader(csvfile, delimiter=',',quotechar='"', quoting=_QUOTINGTYPE)

			for row in reader:
				if row[0] == 'time':
					temp_time = np.array(pf.parseDatetimeList(row[1:]))
					add_values = []
					add_values = (temp_time >= start_date)*(temp_time <= end_date)
					#print(add_values)
					time = np.append(time, np.array(pf.parseDatetimeList(row[1:]))[add_values])
				elif row[0] == 'production':
					production = np.append(production, np.array(pf.parseNumList(row[1:]))[add_values])

	return {'time':time,'production': production}

def getProdAtTimes(series_name, times, file_path = 'Production/'):

	if times.__class__ != np.ndarray:
		times = np.array(times)

	data = loadProductionSeries(series_name, times[0],times[-1] + relativedelta(months = 1), file_path)

	include_indx = np.array([], dtype = bool)
	for t in data['time']:
		if np.count_nonzero(t == times) > 0:
			include_indx = np.append(include_indx, True)
		else:
			include_indx = np.append(include_indx, False)
	#print(include_indx)
	#print(data['time'])
	if len(include_indx) == 1:
		return {'time':data['time'][include_indx][0],'production': data['production'][include_indx][0]}
	else:
		return {'time':data['time'][include_indx],'production': data['production'][include_indx]}


def makeMaxProdFile(data_dir, save_filename):

	max_prod = {}

	for year in os.listdir(data_dir):
		if year == save_filename:
			continue
		year_dir = data_dir + year + '/'
		for month in os.listdir(year_dir):
			month_dir = year_dir + month +'/'
			for filename in os.listdir(month_dir):
				if filename not in max_prod.keys():
					max_prod[filename] = 0
				else:
					data = loadProductionFile(month_dir + filename)

					if np.max(data['production']) > max_prod[filename]:
						max_prod[filename] = np.max(data['production'])


	with open(data_dir + save_filename, 'w', newline='', encoding="latin1") as csvfile:
		datawriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=_QUOTINGTYPE)

		for filename in max_prod.keys():
			datawriter.writerow([filename[:-4], max_prod[filename]])



def loadMaxProdFromFile(file):
	max_prod = {}
	with open(file, 'r', newline='', encoding="latin1") as csvfile:
		reader = csv.reader(csvfile, delimiter=',',quotechar='"', quoting=_QUOTINGTYPE)
		for row in reader:
			max_prod[row[0]] = pf.parseNum(row[1])
	return max_prod

def getMaxProd(loc, start_date, end_date, data_dir = 'Production/'):

    data = getProdByDate(start_date, end_date, locations = loc, load_dir = data_dir)
    
    return data.max()
    







