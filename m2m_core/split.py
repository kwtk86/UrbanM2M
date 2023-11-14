"""run this file to generate tiles for training and testing
it may take 5 minute to generate tiles from an area of 10000km^2"""
import glob
import os, sys
from os.path import join as pj
import numpy as np
import random as rd
from .utils.utils import GDALImage
from tqdm import tqdm
from time import time


__all__ = ['split_main', 'split_check_main']

def is_complete(region: np.ndarray, restriction: np.ndarray):
    if (restriction == 1).sum() >= 0.8 * restriction.size:
        return False
    if (region == 1).sum() == 0:
        return False
    return True


def save_blocks(img_saver: GDALImage, 
                indi_block_dir: str,
                var_name: str,
                arr: np.ndarray,
                start_row: int,
                start_col: int,
                process_func=None
                ) -> None:
    if not os.path.exists(indi_block_dir):
        os.makedirs(indi_block_dir)
    out_path = pj(indi_block_dir, var_name)
    if process_func:
        arr = process_func(arr.copy())
    np.save(out_path, arr)
    # img_saver.save_block(arr, out_path)

"""useless backup start"""
flip_funcs = [lambda x: np.fliplr(x), lambda x: np.flipud(x)]
# 90, 180, 270
rotate_funcs = [lambda x: np.rot90(x), lambda x: np.flipud(x)[:, ::-1], lambda x: np.rot90(x)[::-1, ::-1]]

def splits():
    pass

def crops():
    pass


def loop_blocks(img_saver: GDALImage,
                rasters: dict,
                block_size: int,
                tile_step: int,
                year_range: range,
                region_dir: str = None,
                required_block_count: int = 10000,
                train: bool = True,
                crop:bool = False) -> None:

    for k, v in rasters.items():
        rasters[k] = v[0]
    # spa_vars = [os.path.split(svar)[-1].strip('.tif') for svar in spa_vars]
    region = rasters['range']
    restriction = rasters['restriction']
    if train:
        root_dir = pj(region_dir, f'tile_train')
    else:
        root_dir = pj(region_dir, f'tile_test')
        
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    
    valid_block_count = 0
    existing_block = {}
    max_start_row, max_start_col = region.shape[0], region.shape[1]
    block_rcs = []
    
    if crop:
        while valid_block_count < required_block_count:
            row_start, col_start = rd.randint(block_size, max_start_row), rd.randint(block_size, max_start_col)
            row_end, col_end = row_start + block_size, col_start + block_size
            if f'{row_end}_{col_end}' in existing_block.keys():
                continue
            else:
                existing_block[f'{row_end}_{col_end}'] = 1
            region_block = region[row_start:row_end, col_start:col_end]
            restriction_block = restriction[row_start:row_end, col_start:col_end]
            if is_complete(region_block, restriction_block):  # save current tile or not
                block_rcs.append([row_end, col_end])
                valid_block_count += 1
    else:
        for i, row_end in enumerate(range(tile_step, max_start_row, tile_step)):
            for j, col_end in enumerate(range(tile_step, max_start_col, tile_step)):
                row_start, col_start = row_end - block_size, col_end - block_size
                region_block = region[row_start:row_end, col_start:col_end]
                restriction_block = restriction[row_start:row_end, col_start:col_end]
                if is_complete(region_block, restriction_block):  #  save current tile or not
                    block_rcs.append([row_end, col_end])

    for row_end, col_end in tqdm(block_rcs):
        row_start, col_start = row_end - block_size, col_end - block_size
        flip_rdint, rotate_rdint = rd.randint(0, 1), rd.randint(0, 2)
        process_list = [[pj(root_dir, f'{row_end}_{col_end}'), None],
                            [pj(root_dir, f'{row_end}_{col_end}_f{flip_rdint}'), flip_funcs[flip_rdint]],
                            [pj(root_dir, f'{row_end}_{col_end}_r{flip_rdint}'), rotate_funcs[flip_rdint]]]
        sample_type = rd.randint(1, 3) if crop else 1

        for ras in rasters.keys():
            if 'ras' in ['range', 'restriction']:
                continue
            land_block = rasters[ras][row_start:row_end, col_start:col_end]
            for i in range(sample_type):
                save_blocks(img_saver, process_list[i][0], f'{ras}.npy',
                            land_block, row_start, col_start,process_list[i][1])


def check_binary(ras: np.ndarray, ndv: float, ras_name: str):
    ras_values = np.unique(ras).tolist()
    if ndv !=0 and ndv in ras_values:
        ras_values.pop(ras_values.index(ndv))
    if set(ras_values)!={0,1}:
        raise RuntimeError(f'Invalid raster values in {ras_name}')

def check_normalize(ras: np.ndarray, ndv: float, ras_name: str):
    valid_data = ras[ras!=ndv]
    min_diff = abs(valid_data.min()-0)
    max_diff = abs(valid_data.max()-1)
    valid = (min_diff <= 1e-5 and max_diff <= 1e-5)
    if not valid:
        raise RuntimeError(f'Invalid raster value range in {ras_name}')


def check_rasters(raster_arrays: dict, spa_vars: dict):
    shapes = [array[0].shape for array in raster_arrays.values()]
    # check shape
    if not all(shape == shapes[0] for shape in shapes):
        for name, ras in raster_arrays.items():
            print(f'{name} with a shape of {str(ras[0].shape)}')
        raise RuntimeError('Not all the tifs have the same shape!')
    # check range
    # 对空间变量检查归一化、对其他检查二值化
    for rname, arr in raster_arrays.items():
        if rname in spa_vars.keys():
            check_normalize(arr[0], arr[1], rname)
        else:
            check_binary(arr[0], arr[1], rname)
    # # 检查range.tif的掩膜范围是否最小。
    range_mask = (raster_arrays['range'][0]==1)
    range_mask_sum = range_mask.sum()
    invalid_ras = []
    for rname, arr in raster_arrays.items():
        if rname in ['range', 'restriction']:
            continue
        arr_mask = (arr[0]!=arr[1])
        if arr_mask.sum()<=range_mask_sum:
            invalid_ras.append(rname)
            continue
        mask2 = ((arr_mask==False) & (range_mask==True))
        if mask2.sum() > 0:
            invalid_ras.append(rname)


    if invalid_ras != []:
        for r in invalid_ras:
            print(f'Mask range of {r} is too small')
        raise RuntimeWarning('Invalid mask range')
    return "finish checking, rasters are valid for model training and testing"


def read_rasters(raster_paths: dict) -> dict:
    raster_arrays = {}
    for ras_name, ras in raster_paths.items():
        if ras_name=='restriction':
            if not os.path.exists(ras):
                arr = np.zeros_like(arr)
                ndv = -9999
                print('Failed to find restriction.tif, zero-filled array created')
            else:
                arr, ndv = GDALImage.read_single(ras, -9999, True)
        else:
            arr, ndv = GDALImage.read_single(ras, -9999, True)
        raster_arrays[ras_name] = [arr, ndv]
        print(ras_name)
    return raster_arrays


def get_ras_paths(year_range: range, region_dir: str, spa_vars: dict) -> dict:
    ras = {}
    for varname, varpath in spa_vars.items():
        ras[varname] = varpath
    for year in year_range:
        land_year_name = f'land_{year}.tif'
        land_ras_name = pj(region_dir, 'year', land_year_name)
        ras[f'land_{year}'] = land_ras_name

    ras['range'] = pj(region_dir, 'range.tif')
    ras['restriction'] = pj(region_dir, 'restriction.tif')
    # check existence
    all_exist = 1
    for r in ras.values():
        # ignore restriction
        if r=='restriction':
            continue
        if not os.path.exists(r):
            print(f"{r} does not exist")
            all_exist = 0
    if all_exist == 0:
        raise RuntimeError('0')
    
    return ras


# setting
def split_main(spa_vars0: list = [r'../data-gisa-whn/vars/slope.tif',
                                 r'../data-gisa-whn/vars/county.tif',
                                 r'../data-gisa-whn/vars/town.tif'],
               data_dir: str = r'../data-gisa-whn',
               st_year: int = 2000,
               ed_year: int = 2011,
               is_train: bool = True,
               crop: bool = False,
               tile_size: int = 64,
               tile_step: int = 64,
               ):
    """
    由目标区域完整栅格生成栅格切片
    储存于
        $data_dir/tile_train或
        $data_dir/tile_test
    Args:
        spa_vars0: 使用的空间变量tif文件路径，一般要求位于$data_dir/vars文件夹下，以保证后续数据一致性
        data_dir: 数据根目录
        st_year: 切片起始年份
        ed_year: 切片终止年份。如生成2000-2011的切片，st_year=2000, ed_year=2011
        is_train: 是否生成训练切片
        crop: 预留接口，默认为False。现阶段不用管
        tile_size: 切片大小，默认为64
        tile_step: 相隔多少像素生成一次切片。训练时相隔64，测试时相隔48

    Returns:

    """
    st_year = int(st_year)
    ed_year = int(ed_year)

    assert os.path.exists(data_dir), f'Invalid data directory {data_dir}'
    assert all([svar.endswith('.tif') for svar in spa_vars0]), 'Only .tif supported'
    assert st_year<ed_year, "end year must be larger than start year"

    spa_vars = {}
    for i in range(len(spa_vars0)):
        spa_vars[os.path.basename(spa_vars0[i]).rstrip('.tif')] = spa_vars0[i]

    year_range = range(st_year, ed_year+1)
    img_saver = GDALImage(list(spa_vars.values())[0])
    raster_paths = get_ras_paths(year_range, data_dir, spa_vars)
    rasters = read_rasters(raster_paths)
    check_rasters(rasters, spa_vars)

    loop_blocks(img_saver, rasters, int(tile_size), int(tile_step), year_range, data_dir,
                train = is_train,
                crop = crop)

def split_check_main(spa_vars0: list = [r'D:\zzh2022\15.convlstm\data-gisa-whn/vars/slope.tif',
                                     r'D:\zzh2022\15.convlstm\data-gisa-whn/vars/county.tif',
                                     r'D:\zzh2022\15.convlstm\data-gisa-whn/vars/town.tif'],
                   data_dir: str = r'D:\zzh2022\15.convlstm\data-gisa-whn',
                   st_year: int = 2000,
                   ed_year: int = 2011,
               ):
    """
    由目标区域完整栅格生成栅格切片
    储存于
        $data_dir/tile_train或
        $data_dir/tile_test
    Args:
        spa_vars0: 使用的空间变量tif文件路径，一般要求位于$data_dir/vars文件夹下，以保证后续数据一致性
        data_dir: 数据根目录
        st_year: 切片起始年份
        ed_year: 切片终止年份。如生成2000-2011的切片，st_year=2000, ed_year=2011
        is_train: 是否生成训练切片
        crop: 预留接口，默认为False。现阶段不用管
        tile_size: 切片大小，默认为64
        tile_step: 相隔多少像素生成一次切片。训练时相隔64，测试时相隔48

    Returns:

    """
    st_year = int(st_year)
    ed_year = int(ed_year)

    assert os.path.exists(data_dir), f'Invalid data directory {data_dir}'
    assert all([svar.endswith('.tif') for svar in spa_vars0]), 'Only .tif supported'
    assert st_year<ed_year, "end year must be larger than start year"

    spa_vars = {}
    for i in range(len(spa_vars0)):
        spa_vars[os.path.basename(spa_vars0[i]).rstrip('.tif')] = spa_vars0[i]

    year_range = range(st_year, ed_year+1)
    raster_paths = get_ras_paths(year_range, data_dir, spa_vars)
    rasters = read_rasters(raster_paths)
    checking_info = check_rasters(rasters, spa_vars)
    return checking_info


