import numpy as np
import torch
from torch.utils.data import Dataset
import os, random
from osgeo import gdal
from os.path import join as pj, exists as pex
try:
    from .landscape_config import LandscapeConfig
except:
    pass


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

def is_complete(region: np.ndarray, restriction: np.ndarray):
    if (restriction == 1).sum() >= 0.8 * restriction.size:
        return False
    if (region == 1).sum() == 0:
        return False
    return True


class M2MDatasetBase(Dataset):
    def __init__(self,
                 data_dir: str, inputs: list, spa_vars: list, is_train: bool,
                 sample_count: int = 0, tile_size: int = 64, tile_step: int = 64, start_idx: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.inputs = inputs
        self.spa_vars = [pj(data_dir, 'vars', tif) for tif in spa_vars]
        self.is_train = is_train
        self.sample_count = sample_count
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.start_idx = start_idx

        range_tif = os.path.join(data_dir, 'range.tif')
        assert pex(range_tif), ""
        self.range_arr = open_single_tif(range_tif)

        nonexisting_rasters = []

        for input_tif in self.inputs + self.spa_vars:
            if not pex(input_tif):
                nonexisting_rasters.append(input_tif)

        if nonexisting_rasters:
            err_info = f'Rasters ({",".join(nonexisting_rasters)}) do not exist'
            raise RuntimeError(err_info)
        self.restriction_arr = self.get_restriction_arr()
        self.unique_blocks = self.get_unique_blocks()
        if is_train:
            if sample_count>10:
                self.data = random.sample(self.unique_blocks, sample_count)
            else:
                self.data = self.unique_blocks
        else:
            self.data = self.unique_blocks

        self.spa_arrs = None
        self.land_arrs = None

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
        tile_step, tile_size, start_idx = self.tile_step, self.tile_size, self.start_idx
        max_start_row, max_start_col = region.shape[0], region.shape[1]
        block_rcs = []
        for i, row_end in enumerate(range(tile_step + start_idx, max_start_row, tile_step)):
            for j, col_end in enumerate(range(tile_step + start_idx, max_start_col, tile_step)):
                row_start, col_start = row_end - tile_size, col_end - tile_size
                region_block = region[row_start:row_end, col_start:col_end]
                restriction_block = restriction[row_start:row_end, col_start:col_end]
                if is_complete(region_block, restriction_block):  #  save current tile or not
                    block_rcs.append([row_end, col_end])
        return block_rcs

    def __len__(self):
        return len(self.data)



class CommonDataset(M2MDatasetBase):
    def __init__(self,
                 data_dir: str, inputs: list, spa_vars: list, is_train: bool,
                 sample_count: int = 0, tile_size: int = 64, tile_step: int = 64, start_idx: int = 0):
        super().__init__(data_dir, inputs, spa_vars, is_train, sample_count, tile_size, tile_step, start_idx)

        self.input_arrs = {}
        self.spa_arrs = {}

        for input_tif in self.inputs:
            year = int(os.path.basename(input_tif)[:-4].split('_')[-1])
            self.input_arrs[year] = torch.tensor(open_single_tif(input_tif)).float()

        for spa_tif in self.spa_vars:
            var_name = os.path.basename(spa_tif)[:-4]
            self.spa_arrs[var_name] = torch.tensor(open_single_tif(spa_tif)).float()

    def __getitem__(self, idx):
        tile_size = self.tile_size
        row_end, col_end = self.data[idx]
        spa_var_tensor = torch.empty(len(self.spa_vars), tile_size, tile_size)
        for i, (k, spa_arr) in enumerate(self.spa_arrs.items()):
            spa_var_tensor[i, :, :] = spa_arr[row_end - tile_size: row_end,
                                      col_end - tile_size: col_end]


        input_tensor = torch.empty(len(self.inputs), 1, tile_size, tile_size).float()
        for i, (year, input_arr) in enumerate(self.input_arrs.items()):
            input_tensor[i, 0, :, :] = input_arr[row_end - tile_size: row_end,
                                       col_end - tile_size: col_end]
        return f'{row_end}_{col_end}', spa_var_tensor, input_tensor




class DatasetWithLandscape(M2MDatasetBase):
    def __init__(self,
                 landscapes: list,
                 data_dir: str,
                 inputs: list,
                 spa_vars: list,
                 is_train: bool,
                 sample_count: int = 0,
                 tile_size: int = 64,
                 tile_step: int = 64,
                 start_idx: int = 0,
                 ):
        super().__init__(data_dir, inputs, spa_vars, is_train, sample_count, tile_size, tile_step, start_idx)
        self.landscaper = LandscapeConfig(landscapes)
        self.land_arrs = {}
        self.spa_arrs = {}
        # self.landscape_arrs = {}

        for input_tif in self.inputs:
            year = int(os.path.basename(input_tif)[:-4].split('_')[-1])
            year_arr = open_single_tif(input_tif)
            self.land_arrs[year] = torch.tensor(year_arr).float()
            # self.landscape_arrs[year] = self.landscaper.get_landscape(year_arr).float()

        for spa_tif in self.spa_vars:
            var_name = os.path.basename(spa_tif)[:-4]
            self.spa_arrs[var_name] = torch.tensor(open_single_tif(spa_tif)).float()

    def __getitem__(self, idx):
        tile_size = self.tile_size
        row_end, col_end = self.data[idx]
        spa_var_tensor = torch.empty(len(self.spa_vars), tile_size, tile_size)
        for i, (k, spa_arr) in enumerate(self.spa_arrs.items()):
            spa_var_tensor[i, :, :] = spa_arr[row_end - tile_size: row_end,
                                              col_end - tile_size: col_end]


        land_tensor = torch.empty(len(self.inputs), 1 + self.landscaper.types,
                                  tile_size, tile_size).float()
        for i, (year, land_arr) in enumerate(self.land_arrs.items()):
            land_tile = land_arr[row_end - tile_size: row_end, col_end - tile_size: col_end]
            land_tensor[i, 0, :, :] = land_tile
            land_tensor[i, 1:, :, :] = self.landscaper.get_landscape(land_tile.detach().numpy())
            # for i, (year, landscape_arr) in enumerate(self.landscape_arrs.items()):
        #     land_tensor[i, 1:, :, :] = landscape_arr[:,
        #                                row_end - tile_size: row_end,
        #                                col_end - tile_size: col_end]




        return f'{row_end}_{col_end}', spa_var_tensor, land_tensor





