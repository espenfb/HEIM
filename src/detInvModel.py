# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:07:41 2017

@author: espenfb
"""

import pyomo.environ as pe
import os
import detModelRes as dmr
import systemData as sd
import pandas as pd
from dateutil.relativedelta import relativedelta
import time
import detInvData as did
import detInvFormulation as dif
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

GW2MW = 1000


class deterministicModel(object):
    ''' Deterministic investment model for regional power system
     with hydogen loads. '''

    def __init__(self, time_data, dirs, mutables={}):

        # Times
        for k in time_data.keys():
            setattr(self, k, time_data[k])

        # Directories
        for k in dirs.keys():
            setattr(self, k, dirs[k])

        self.mutables = mutables
        # Import system data
        self.data = sd.systemData(dirs)
        wind = self.data.wind_series
        solar = self.data.solar_series

        self.time = pd.date_range(start=self.start_date,
                                  end=self.end_date - relativedelta(hour=1),
                                  freq='H')

        self.time = self.time[self.time.isin(self.data.load_series.index)]
        self.time = self.time[self.time.isin(wind.index)]
        self.time = self.time[self.time.isin(solar.index)]

        self.timerange = range(len(self.time))

    def buildModel(self):

        print('Building deterministic investment model...')
        self.detModel = dif.buildDetModel(mutables=self.mutables)

        # Create concrete instance
        self.detDataInstance = did.detData(self)
        print('Creating LP problem instance...')
        self.instance = self.detModel.create_instance(
                                data=self.detDataInstance,
                                name="Deterministic operation model",
                                namespace='detData')

        # Enable access to duals
        self.instance.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    def solve(self, printOutput=True):

        # Connect to solver
        self.opt = pe.SolverFactory('cplex')  #, solver_io='python')

        if printOutput:
            print('Solving deterministic operation model...')

        # Solve model
        start_time = time.time()
        self.pyomo_res = \
            self.opt.solve(self.instance) ##,
        #                   tee=printOutput)  # stream the solver output
        #                   keepfiles=False,  # print the LP file - debugging
        #                   symbolic_solver_labels=True,
        #                   warmstart=True,
        #                   options={"Method": 2,
        #                            "Crossover": 0,
        #                            "BarHomogeneous": 1})
        #                       "QCPDual":0})#,
        #                       "NodeMethod": 2,
        #                       "MIPGap": 0.01,
        #                       "MIPFocus": 3)

        # self.instance.write('model.mps',
        #                     io_options={'symbolic_solver_labels':True})

        self.solution_time = time.time()-start_time

    def printModel(self, name='invModel.txt'):

        self.instance.pprint(name)

    def processResults(self, printOutput=True):
        ''' Prosessing results from pyomo form to pandas data-frames
        for storing and plotting. '''

        if printOutput:
            print('Prosessing results from deteministic model...')

        model = self.instance

        dmr.processRes(self, model)

    def saveRes(self, save_dir):
        ''' Saving prosessed results.  '''

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        sol_time = pd.DataFrame.from_dict({'sol_time': [self.solution_time]})
        sol_time.to_csv(save_dir + 'sol_time.csv')

        dmr.saveRes(self, save_dir)

    def importResults(self, import_dir):
        ''' Importing results from files. '''

        dmr.importRes(self, import_dir)
