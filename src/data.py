import pandas as pd 
import numpy as np 
import pickle as pkl
import warnings
from math import fabs, ceil
from sklearn.preprocessing import StandardScaler

import utils
from settings import libPath

@utils.timeit
def get_data(con, query, index=None):
    data = pd.read_sql(query, con)
    data.astype(np.float32)
    if index is not None:
        data.set_index(index, inplace=True)
    return data

class TrainingData():
    def __init__(self, data_reader, target, data_split=None):
        self.data_ = data_reader.get_data()
        self.nobs_ = self.data_.values.shape[0]
        self.features_ = [x for x in list(self.data_.columns) if x !=target]
        self.target_ = target
        
        # initialize our dataset
        if data_split is not None:
            self.shuffle()
            self.standardize()
            self.split(*data_split)

    def shuffle(self):
        self.data_ = self.data_.sample(frac=1).reset_index(drop=True)

    def standardize(self):
        cols = self.data_.columns
        self.data_ = pd.DataFrame(data=StandardScaler().fit_transform(self.data_), columns=cols) 

    def split(self, train_frac, valid_frac, test_frac):
        epsi = 1e-10
        total_frac = train_frac + valid_frac + test_frac
        if total_frac > 1:
            raise ValueError('Split fractions must sum to a max of 1. Current sum is: %s' % str(train_frac + valid_frac + test_frac))
        elif fabs(1-total_frac) > epsi :
            warnings.warn('Currently not using entire dataset. Total fraction of data used is: %s'% total_frac)
        
        nobs = self.data_.shape[0]
        
        n_train_obs = ceil(nobs * train_frac)
        n_valid_obs = ceil(nobs * valid_frac)

        train_df = self.data_.iloc[0:n_train_obs]
        self.train_x = train_df[self.features_]
        self.train_y = train_df[self.target_]

        valid_df =  self.data_.iloc[n_train_obs:n_train_obs + n_valid_obs]
        self.valid_x = valid_df[self.features_]
        self.valid_y = valid_df[self.target_]

        test_df = self.data_.iloc[n_train_obs + n_valid_obs:]
        self.test_x = test_df[self.features_]
        self.test_y = test_df[self.target_]

class DataReader():
    def __init__(self, source='sql'):
        self.source_ = source
        self.data_ = pd.DataFrame() 

    @utils.timeit
    def fetch_data(self, **kwargs):
        if self.source_ == 'sql':
            if 'index' in kwargs.keys():
                self.__fetch_sql(kwargs['con'], kwargs['query'], kwargs['index'])
            else:
                self.__fetch_sql(kwargs['con'], kwargs['query'])

        elif self.source_ == 'file':
            self.__fetch_file(kwargs['file'])
    
    def __fetch_sql(self,con, query, index=None):
        data = pd.read_sql(query, con)
        data.astype(np.float32)
        if index is not None:
            data.set_index(index, inplace=True)
        self.data_ = data
    
    def __fetch_file(self, path):
        # this is file type dependent, assumed to be .pkl
        path = libPath + '\\data\\' + path
        with open(path, 'rb') as f:
            data = pkl.load(f)
        self.data_ = data 
    
    def get_data(self):
        return self.data_.copy(deep=True)
        