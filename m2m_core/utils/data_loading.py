import numpy as np
import torch


# from config import *
from torch.utils.data import Dataset
import os, random, glob
from osgeo import gdal
from os.path import join as pj, exists as pex
from collections import OrderedDict as odict


def open_single_tif(path: str) -> np.array:

    dataset = gdal.Open(path)
    band_count = dataset.RasterCount
    bands = []
    for i in range(1, band_count + 1):
        band = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
        bands.append(band)
    if len(bands) > 1:
        bands = np.stack(bands)
    else:
        bands = bands[0]
    return bands

class TrainDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 inputs: list,
                 spa_vars: list,
                 sample_count: int,
                 height: int = 64):
        self.data_dir = data_dir
        self.inputs  = inputs
        self.spa_vars = spa_vars
        self.raw_dirs = os.listdir(data_dir)
        self.height = height
        self.unique_blocks = self.get_unique_blocks()

        self.inputs = [yinput.replace('.tif', '.npy') for yinput in self.inputs]
        self.spa_vars = [spa_var.replace('.tif', '.npy') for spa_var in self.spa_vars]


        if sample_count>10:
            self.sampled_blocks = random.sample(self.unique_blocks, sample_count)
        else:
            self.sampled_blocks = self.unique_blocks
        self.data = [os.path.join(data_dir, item) for item in self.sampled_blocks]

    def get_unique_blocks(self):
        blocks = ["_".join(ddir.split('_')[:2]) for ddir in self.raw_dirs]
        unique_blocks = list(set(blocks))
        return unique_blocks

    def __getitem__(self, idx):
        input_tensor   = torch.empty(len(self.inputs), 1, self.height, self.height).float()
        block = self.data[idx]
        rc = os.path.basename(block)

        # spatial variables
        if self.spa_vars:
            spa_var_tensor = torch.empty(len(self.spa_vars), self.height, self.height)
            for i, variable in enumerate(self.spa_vars):
                var_array = np.load(pj(block, variable), allow_pickle=True)
                var_tensor = torch.as_tensor(var_array).float()
                spa_var_tensor[i, :, :] = var_tensor
        else:
            spa_var_tensor = []
        # read by year
        for i, year in enumerate(self.inputs):
            land_arr = np.load(pj(block, year))
            land_tensor = torch.as_tensor(land_arr).float()
            input_tensor[i, 0, :, :]  = land_tensor

        return rc, spa_var_tensor, input_tensor

    def __len__(self):
        return len(self.data)

def is_complete(region: np.ndarray, restriction: np.ndarray):
    if (restriction == 1).sum() >= 0.8 * restriction.size:
        return False
    if (region == 1).sum() == 0:
        return False
    return True


class Dataset_woSplit(Dataset):
    def __init__(self,
                 data_dir: str,
                 inputs: list,
                 spa_vars: list,
                 is_train: bool,
                 sample_count: int = 0,
                 tile_size: int = 64,
                 tile_step: int = 64,):
        self.data_dir = data_dir
        range_tif = os.path.join(data_dir, 'range.tif')
        assert pex(range_tif), ""
        self.range_arr = open_single_tif(range_tif)
        self.inputs   = [pj(data_dir, 'year', tif) for tif in inputs]
        self.spa_vars = [pj(data_dir, 'vars', tif) for tif in spa_vars]
        nonexisting_rasters = []
        for input_tif in self.inputs + self.spa_vars:
            if not pex(input_tif):
                nonexisting_rasters.append(input_tif)
        if nonexisting_rasters:
            err_info = f'Rasters ({",".join(nonexisting_rasters)}) do not exist'
            raise RuntimeError(err_info)

        self.input_arrs = {}
        self.spa_arrs = {}
        for input_tif in self.inputs:
            year = int(os.path.basename(input_tif)[:-4].split('_')[-1])
            self.input_arrs[year] = torch.tensor(open_single_tif(input_tif)).float()
        for spa_tif in self.spa_vars:
            var_name = os.path.basename(spa_tif)[:-4]
            self.spa_arrs[var_name] = torch.tensor(open_single_tif(spa_tif)).float()
        self.restriction_arr = self.get_restriction_arr()
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.unique_blocks = self.get_unique_blocks()
        if is_train:
            if sample_count>10:
                self.sampled_blocks = random.sample(self.unique_blocks, sample_count)
            else:
                self.sampled_blocks = self.unique_blocks
            self.data = self.sampled_blocks
        else:
            self.data = self.unique_blocks

    def get_restriction_arr(self):
        if pex(pj(self.data_dir, 'restriction.tif')):
            arr = open_single_tif(pj(self.data_dir, 'restriction.tif'))
        else:
            print('Failed to find restriction.tif, zero-filled array created')
            arr = np.zeros_like(self.range_arr)
        return arr


    def get_unique_blocks(self):
        region = self.range_arr
        restriction = self.restriction_arr
        tile_step, tile_size = self.tile_step, self.tile_size
        max_start_row, max_start_col = region.shape[0], region.shape[1]
        block_rcs = []
        for i, row_end in enumerate(range(tile_step, max_start_row, tile_step)):
            for j, col_end in enumerate(range(tile_step, max_start_col, tile_step)):
                row_start, col_start = row_end - tile_size, col_end - tile_size
                region_block = region[row_start:row_end, col_start:col_end]
                restriction_block = restriction[row_start:row_end, col_start:col_end]
                if is_complete(region_block, restriction_block):  #  save current tile or not
                    block_rcs.append([row_end, col_end])
        return block_rcs

    def __getitem__(self, idx):
        tile_size = self.tile_size
        row_end, col_end = self.data[idx]
        if self.spa_vars:
            spa_var_tensor = torch.empty(len(self.spa_vars), tile_size, tile_size)
            for i, (k, spa_arr) in enumerate(self.spa_arrs.items()):
                spa_var_tensor[i, :, :] = spa_arr[row_end - tile_size: row_end,
                                                  col_end - tile_size: col_end]
        else:
            spa_var_tensor = []
        input_tensor   = torch.empty(len(self.inputs), 1, tile_size, tile_size).float()
        for i, (year, input_arr) in enumerate(self.input_arrs.items()):
            input_tensor[i, 0, :, :]  = input_arr[row_end - tile_size: row_end,
                                                  col_end - tile_size: col_end]
        return f'{row_end}_{col_end}', spa_var_tensor, input_tensor

    def __len__(self):
        return len(self.data)


# class TestDataset_woSplit(Dataset):
#     def __init__(self,
#                  data_dir: str,
#                  inputs: list,
#                  spa_vars: list,
#                  sample_count: int,
#                  tile_size: int = 64,
#                  tile_step: int = 48,):
#
#         range_tif = os.path.join(data_dir, 'range.tif')
#         assert pex(range_tif), ""
#         self.range_arr = open_single_tif(range_tif)
#         self.inputs   = [pj(data_dir, 'year', tif) for tif in inputs]
#         self.spa_vars = [pj(data_dir, 'vars', tif) for tif in spa_vars]
#         nonexisting_rasters = []
#         for input_tif in self.inputs + self.spa_vars:
#             if not pex(input_tif):
#                 nonexisting_rasters.append(input_tif)
#         if nonexisting_rasters:
#             err_info = f'Rasters ({",".join(nonexisting_rasters)}) do not exist'
#             raise RuntimeError(err_info)
#
#         self.input_arrs = {}
#         self.spa_arrs = {}
#         for input_tif in self.inputs:
#             year = int(os.path.basename(input_tif)[:-4].split('_')[-1])
#             self.input_arrs[year] = torch.tensor(open_single_tif(input_tif)).float()
#         for spa_tif in self.spa_vars:
#             var_name = os.path.basename(spa_tif)[:-4]
#             self.spa_arrs[var_name] = torch.tensor(open_single_tif(spa_tif)).float()
#         self.restriction_arr = self.get_restriction_arr()
#         self.tile_size = tile_size
#         self.tile_step = tile_step
#         self.unique_blocks = self.get_unique_blocks()
#         self.data = self.unique_blocks
#
#     def get_restriction_arr(self):
#         if pex(pj(self.data_dir, 'restriction.tif')):
#             arr = open_single_tif(pj(self.data_dir, 'restriction.tif'))
#         else:
#             print('Failed to find restriction.tif, zero-filled array created')
#             arr = np.zeros_like(self.range_arr)
#         return arr
#
#
#     def get_unique_blocks(self):
#         region = self.range_arr
#         restriction = self.restriction_arr
#         tile_step, tile_size = self.tile_step, self.tile_size
#         max_start_row, max_start_col = region.shape[0], region.shape[1]
#         block_rcs = []
#         for i, row_end in enumerate(range(tile_step, max_start_row, tile_step)):
#             for j, col_end in enumerate(range(tile_step, max_start_col, tile_step)):
#                 row_start, col_start = row_end - tile_size, col_end - tile_size
#                 region_block = region[row_start:row_end, col_start:col_end]
#                 restriction_block = restriction[row_start:row_end, col_start:col_end]
#                 if is_complete(region_block, restriction_block):  #  save current tile or not
#                     block_rcs.append([row_end, col_end])
#         return block_rcs
#
#     def __getitem__(self, idx):
#         tile_size = self.tile_size
#         row_end, col_end = self.data[idx]
#         if self.spa_vars:
#             spa_var_tensor = torch.empty(len(self.spa_vars), tile_size, tile_size)
#             for i, (k, spa_arr) in enumerate(self.spa_arrs.items()):
#                 spa_var_tensor[i, :, :] = spa_arr[row_end - tile_size: row_end,
#                                                   col_end - tile_size: col_end]
#         else:
#             spa_var_tensor = []
#         input_tensor   = torch.empty(len(self.inputs), 1, tile_size, tile_size).float()
#         for i, (year, input_arr) in enumerate(self.input_arrs.items()):
#             input_tensor[i, 0, :, :]  = input_arr[row_end - tile_size: row_end,
#                                                   col_end - tile_size: col_end]
#         return f'{row_end}_{col_end}', spa_var_tensor, input_tensor
#
#     def __len__(self):
#         return len(self.data)