# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:01:36 2017

@author: espenfb
"""

def timeVarToDict(model, var, new_indx):
    var_values = var.get_values()
    return {i: [var_values[t,i] for t in model.TIME] for i in new_indx}

def simParamToDict(model, var, new_indx):
    param_values = var.extract_values()
    return {i: [param_values[t,i] for t in model.TIME_SIM] for i in new_indx}

def planVarToDict(model, var, new_indx):
    var_values = var.get_values()
    return {i: [var_values[t,i] for t in model.TIME_PLAN] for i in new_indx}

def timeParamToDict(model, var, new_indx):
    param_values = var.extract_values()
    return {i: [param_values[t,i] for t in model.TIME] for i in new_indx}

def timeDualToDict(model, constr, new_indx):
    constr_dict = dict(constr)
    return { i : [model.dual[constr_dict[t,i]] for t in model.TIME] for i in new_indx}



def timeVarToDict_TS(model, var, new_indx, node):
    var_values = var.get_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.TIME:
                out[i,n].append(var_values[t,i,n])
    return out

def timeParamToDict_TS(model, var, new_indx, node):
    param_values = var.extract_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.TIME:
                out[i,n].append(param_values[t,i,n])
    return out

def timeDualToDict_TS(model, constr, new_indx, node):
    constr_dict = dict(constr)
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.TIME:
                out[i,n].append(model.dual[constr_dict[t,i,n]])
    return out

# V2
def timeVarToDict_V2(model, var, new_indx, node):
    var_values = var.get_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.NODE_TO_TIME[n]:
                out[i,n].append(var_values[n,t,i])
    return out

def timeParamToDict_V2(model, var, new_indx, node):
    param_values = var.extract_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.NODE_TO_TIME[n]:
                out[i,n].append(param_values[n,t,i])
    return out

def timeDualToDict_V2(model, constr, new_indx, node):
    constr_dict = dict(constr)
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.NODE_TO_TIME[n]:
                out[i,n].append(model.dual[constr_dict[n,t,i]])
    return out

def timeDualToDict_line(model, constr, new_indx, node):
    constr_dict = dict(constr)
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.NODE_TO_TIME[n]:
                out[i,n].append(model.dual[constr_dict[n,t,i[0],i[1]]])
    return out

def timeVarToDict_resPenalty(model, var, new_indx, node):
    var_values = var.get_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            for t in model.FIX_RESERVOIR_TIME:
                if t in model.NODE_TO_TIME[n]:
                    out[i,n].append(var_values[n,t,i])
    return out

def timeVarToDict_endSto(model, var, new_indx, node):
    var_values = var.get_values()
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            t = model.TIME_PLAN[-1]
            if t in model.NODE_TO_TIME[n]:
                out[i,n].append(var_values[n,t,i])
    return out

def timeDualToDict_endSto(model, constr, new_indx, node):
    constr_dict = dict(constr)
    out = {}
    for i in new_indx:
        for n in node:
            out[i,n] = []
            t = model.TIME_PLAN[-1]
            out[i,n].append(model.dual[constr_dict[n,t,i]])
    return out

