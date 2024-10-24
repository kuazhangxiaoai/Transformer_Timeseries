import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import data_formatters.utils as utils
from conf.conf import Conf
from pathlib import Path
import pickle

class HotelDataset(Dataset):
    def __init__(self, cnf, data_formatter):
        self.params = cnf.all_params
        self.csv = utils.data_csv_path(cnf.ds_name)
        self.data = pd.read_csv(self.csv, index_col=0, na_filter=False)
        self.data = data_formatter.category_filter(self.data, self.params['category'])
        self.data = data_formatter.share_amount_filter(self.data)
        self.data = data_formatter.status_filter(self.data, self.params['status'])
        self.csv_dir = os.path.join(Path(self.csv).parent)
        self.cache = os.path.join(self.csv_dir, 'cache.pkl')


        if not os.path.exists(self.cache):
            self.trainset, self.validset, self.testset = data_formatter.split_data(
                self.data,
                self.params['train_begin'],
                self.params['train_end'],
                self.params['val_begin'],
                self.params['val_end'],
                self.params['test_begin'],
                self.params['test_end']
            )
            x = {
                'trainset': self.trainset,
                'valset': self.validset,
                'testset': self.testset
            }
            with open(self.cache, 'wb') as file:
                pickle.dump(x,file)
        else:
            with open(self.cache, 'rb') as file:
                cache = pickle.load(file)
                self.trainset = cache['trainset']
                self.valset = cache['valset']
                self.testset = cache['testset']

    def preprocess(self):

        return




if __name__ == '__main__':
    cnf = Conf(
        conf_file_path='../conf/hotel.yaml',
        seed=15,
        exp_name='hotel',
        log=True
    )
    data_formatter = utils.make_data_formatter(cnf.ds_name)
    dataset = HotelDataset(cnf, data_formatter)