# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:29:04 2017

@author: Espen Flo Bødal
"""
import datetime


def parseNum(num,default=None):
    '''parse number and return a float'''
    if default is None:
        return float(num)
    elif num=='' or num is None:
        return default
    else:
        return float(num)

def parseNumList(num_list):
    temp_list = []
    for val in num_list:
        temp_list.append(parseNum(val))
    return temp_list

def parseInt(num,default=None):
    '''parse number and return a int'''
    if default is None:
        return int(num)
    elif num=='' or num is None:
        return default
    else:
        return int(num)
    
def parseInt2(num,default=None):
    '''parse string and return int or none'''
    alpha = any(c.isalpha() for c in num)
    if num=='' or num is None or alpha:
        return default
    else:
        return int(num)

def parseIntList(num_list):
    temp_list = []
    for val in num_list:
        temp_list.append(parseInt(val))

    return temp_list


def parseDatetime(datetime_str):
    year = parseInt(datetime_str[0:4])
    month = parseInt(datetime_str[5:7])
    day = parseInt(datetime_str[8:10])
    if len(datetime_str) >= 11:
        hour = parseInt(datetime_str[11:13])
    else:
        return datetime.datetime(year,month,day)
    if len(datetime_str) >= 14:
        minute = parseInt(datetime_str[14:16])
    else:
        return datetime.datetime(year,month,day,hour)
    if len(datetime_str) >= 17:
        second = parseInt(datetime_str[17:19])
    else:
        return datetime.datetime(year,month,day,hour,minute)

    return datetime.datetime(year,month,day,hour,minute,second)

def parseDatetimeList(datetime_list):
    res_list = []
    for i in range(len(datetime_list)):
        res_list.append(parseDatetime(datetime_list[i]))
    return res_list


def parseDate(date_str):
    day = parseInt(date_str[0:2])
    month = parseInt(date_str[3:5])
    year = parseInt(date_str[6:10])
    hour = parseInt(date_str[11:13])
    minute = parseInt(date_str[14:16])

    return datetime.datetime(year,month,day,hour,minute, tzinfo = datetime.timezone.utc)


def representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
    
def parseLineTuple(line):
    out = []
    temp = ''
    for i in line:
        if RepresentsInt(i):
            temp += i
        elif temp is not '':
            out.append(int(temp))
            temp = ''
    if temp is not'':
        out.append(int(temp))
    return tuple(out)
