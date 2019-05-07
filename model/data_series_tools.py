# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:50:39 2017

@author: espenfb
"""

from scipy import interpolate
import numpy as np


def scaleSeries(series, factor):
	"""Used to scale weekly or daily series to hourly series.
	Eks: weekly to hourly series -> factor = 168 """
	tempSeries = series[:]
	scSeries = [0]*len(tempSeries)*factor
	for row in range(len(tempSeries)):
		for exp_row in range(factor):
			scSeries[row*factor+exp_row] = tempSeries[row]*(1/factor)
	return scSeries


def splinalInterpolation(series,expFac,appendNum=24):
	tempSeries = series[:]
	tempSeries.append(series[len(series)-1])
	x = np.arange(0,len(tempSeries)*expFac,expFac)
	tck = interpolate.splrep(x,tempSeries)
	xnew = np.arange(0,len(tempSeries)*expFac,1)
	yinter = interpolate.splev(xnew,tck,der=0)
	ynew = yinter*(1/yinter.sum())

	return ynew[0:len(series)*expFac+appendNum]


def expandSeries(series,num):
	"""Adds last element of original series to end of expanded series
	"num" times. Used to expand to short series, Ex: 7*52 = 364 -> 365. """
	length = len(series)
	for i in range(num):
		series.append(series[length-1])
	return series

def appendSeries(series,num_app,num_last):
	"""Adds last num_last elements of the series num_app times."""
	temp_series = series[:]
	length = len(series)
	last_list = series[length-num_last:length]
	for i in range(num_app):
		for j in range(num_last):
			temp_series.append(last_list[j])
	return temp_series


