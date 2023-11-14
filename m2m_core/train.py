import os, sys
from os.path import (exists as pex, join as pj)
from .config import *
from .model import CC_ConvLSTM
from .utils.data_loading import TrainDataset, Dataset_woSplit
from .utils.trainers.trainer import Trainer
import argparse
from datetime import datetime
import glob

from collections import defaultdict

class Namespace:
    def __init__(self):
        self._data = defaultdict(Namespace)

    def __getattr__(self, name):
        return self._data[name]

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

__all__ = ['train_main', 'train_model']



def check_args(args):

    args.input_tifs   = [f'land_{year}.tif' for year in range(args.start_year, args.start_year + args.in_len + args.out_len, 1)]
    args.band = 1 + len(args.spa_var_tifs)

    formatted_date = datetime.now().strftime("%m_%d_%H_%M")
    args.model_name = f'convlstm-t{formatted_date}'
    args.use_ce = True
    args.use_mix = True
    return args



# def assert_args(args):
#     assert args.sample_count <= len(os.listdir(args.tile_dir)), "sample count smaller than valid tile count"

def train_main(start_year:   int,
               in_len:       int,
               out_len:      int,
               data_dir:     str,
               spa_vars:     list,
               batch_size:   int,
               lr:           float,
               sample_count: int,
               val_prop:     float):

    assert 0<val_prop<1, \
        "Invalid val_prop, required 0<val_prop<1"
    assert 0<lr<1, \
        "Invalid lr, required 0<lr<1"

    assert start_year>1900 and round(start_year)==start_year, \
        "Invalid start_year, required start_year>1900 and being integer"
    assert in_len>0 and round(in_len)==in_len, \
        "Invalid in_len, required in_len>0 and being integer"
    assert out_len>0 and round(out_len)==out_len, \
        "Invalid out_len, required out_len>0 and being integer"

    assert batch_size>0 and round(batch_size)==batch_size, \
        "Invalid batch_size, required batch_size>1900 and being integer"
    assert sample_count>100 and round(sample_count)==sample_count, \
        "Invalid sample_count, required sample_count>100 and being integer"

    edirs = [data_dir]
    for edir in edirs:
        if not os.path.exists(edir):
            raise RuntimeError(f'{edir} does not exist')

    start_year = int(start_year)
    in_len = int(in_len)
    out_len = int(out_len)
    sample_count = int(sample_count)
    batch_size = int(batch_size)



    args = Namespace()

    args.in_len = in_len
    args.out_len = out_len
    args.start_year = start_year
    # args.tile_dir = os.path.join(data_dir, 'tile_train')
    args.tile_dir = os.path.join(data_dir)
    args.spa_var_tifs = spa_vars
    args.tile_size = 64

    args.batch_size = batch_size
    args.lr = lr
    args.sample_count = sample_count
    args.val_prop = val_prop
    ##########
    args.height = 64
    args.nlayers = 1
    args.filter_size = 5
    args.epochs = 999
    args.eta_decay = 0.015
    args.model_dir = 'trained_models'
    args.num_workers = 0

    args = check_args(args)

    # assert_args(args)
    train_model(args)

def train_model(args):
    model = CC_ConvLSTM.CC_ConvLSTM(input_chans  = args.band,
                                    output_chans = 1,
                                    hidden_size  = args.tile_size,
                                    filter_size  = args.filter_size,
                                    num_layers   = args.nlayers,
                                    img_size     = args.tile_size,
                                    in_len      = args.in_len,
                                    out_len     = args.out_len)
    model = model.cuda()
    dataset = Dataset_woSplit(args.tile_dir,
                              args.input_tifs,
                              args.spa_var_tifs,
                              True,
                              args.sample_count,
                              args.tile_size,
                              args.tile_size)
    t = Trainer(model, args, dataset, device)
    t.loop()

