from .config import *

import argparse
import os, sys
from typing import Literal
from os.path import join as pj
from .utils.data_loading import CommonDataset
# from .utils.landscape_config import PatchSize
from .utils.future_land import *
from .utils.trainers.tester import *
from .model import CC_ConvLSTM
from .prob2sim import prob2sim_main

__all__ = ['test_main', 'test_autoreg_main']



def check_args(args):

    args.range_tif = os.path.join(args.data_root_dir, 'range.tif')

    
    args.input_tifs = [os.path.join(args.data_root_dir, 'year', f'land_{year}.tif') for year
                       in range(args.start_year, args.start_year + args.in_len, 1)]

    args.prob_tifs  = [os.path.join(args.prob_dir, f'prob_{year}.tif') for year in 
                       range(args.start_year + args.in_len, args.start_year + args.in_len + args.out_len)]
    args.use_mix = True

    return args



def test_main(start_year: int,
              fisrt_sim_year: int,
              out_len: int,
              data_dir: str,
              model_path: str,
              prob_dir: str,
              tile_size: int,
              tile_step: int,
              tile_start_idx: int = 0,
              batch_size: int = 128,
              num_workers: int = 0,
              strategy: Literal['mean', 'max'] = 'mean',
              dataset_type: Literal['default', 'landscape'] = 'default'):
    """
    案例：使用2006-2011年作为输入，生成2012-2017的转换概率图
    Args:
        start_year:     输入数据的第一个年份，案例中为2006
        fisrt_sim_year: 模拟的第一个年份，案例中为2012
        out_len:        模拟的年份数目，案例中为2017-2012+1=6
        data_dir:       数据根目录
        model_path:     模型文件路径
        prob_dir:       输出结果保存路径
        batch_size:     批量大小
        num_workers:    线程数

    Returns:

    """


    assert start_year>1900 and round(start_year)==start_year, \
        "Invalid start_year, required start_year>1900 and being integer"
    assert fisrt_sim_year>1900 and round(fisrt_sim_year)==fisrt_sim_year, \
        "Invalid first_year_tosim, required first_year_tosim>1900 and first_year_tosim integer"
    assert out_len>0 and round(out_len)==out_len, \
        "Invalid out_len, required out_len>0 and being integer"
    assert batch_size>0 and round(batch_size)==batch_size, \
        "Invalid batch_size, required batch_size>1900 and being integer"
    assert num_workers>=0 and round(num_workers)==num_workers, \
        "Invalid num_workers, required num_workers>0 and being integer"

    edirs = [data_dir, pj(data_dir, 'range.tif'), model_path]
    for edir in edirs:
        if not os.path.exists(edir):
            raise RuntimeError(f'{edir} does not exist')

    start_year = int(start_year)
    fisrt_sim_year = int(fisrt_sim_year)
    out_len = int(out_len)
    batch_size = int(batch_size)
    num_workers = int(num_workers)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.start_year = start_year
    args.in_len = fisrt_sim_year - start_year
    args.out_len = out_len
    args.data_root_dir = data_dir

    args.model_path = model_path
    args.prob_dir = prob_dir
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.tile_size = tile_size
    args.tile_step = tile_step
    args.tile_start_idx = tile_start_idx

    # args.height = 64
    # args.tile_step = 48
    args.edge_width = 4


    args = check_args(args)
    args.autoreg = False
    args.strategy = strategy
    args.dataset_type = dataset_type
    test_model(args)


def test_autoreg_main(start_year: int,
                      first_simed_year:int,
                      out_len: int,
                      data_dir: str,
                      model_path: str,
                      sim_dir: str,
                      prob_dir: str,
                      demands: list,
                      tile_size: int = 64,
                      tile_step: int = 48,
                      strategy: str = 'mean',
                      batch_size: int = 128,
                      num_workers: int = 0,
                      dataset_type: Literal['default', 'landscape'] = 'default'):

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if not os.path.exists(prob_dir):
        os.makedirs(prob_dir)

    assert len(demands) == out_len, ""
    assert first_simed_year>1900 and round(first_simed_year) == first_simed_year,\
        "Invalid first simulated year, required being integer"
    assert out_len>0 and round(out_len)==out_len, \
        "Invalid out_len, required out_len>0 and being integer"

    assert batch_size>0 and round(batch_size)==batch_size, \
        "Invalid batch_size, required batch_size>1900 and being integer"
    assert num_workers>=0 and round(num_workers)==num_workers, \
        "Invalid num_workers, required num_workers>0 and being integer"
    assert start_year>1900 and round(start_year)==start_year, \
        "Invalid start_year, required start_year>1900 and being integer"
    edirs = [data_dir, pj(data_dir, 'range.tif'), model_path]
    for edir in edirs:
        if not os.path.exists(edir):
            raise RuntimeError(f'{edir} does not exist')

    start_year = int(start_year)
    first_simed_year = int(first_simed_year)
    out_len = int(out_len)
    batch_size = int(batch_size)
    num_workers = int(num_workers)

    year_idx = 0
    for year in range(start_year, start_year + out_len):
        test_year_autoreg(year,
                          first_simed_year,
                          year + out_len,
                          1,
                          data_dir,
                          model_path,
                          sim_dir,
                          prob_dir,
                          tile_size,
                          tile_step,
                          # 32 * (year_idx % 2)
                          0,
                          strategy,
                          batch_size,
                          num_workers,
                          dataset_type)

        if start_year == year:
            final_gt_tif = os.path.join(data_dir,
                                        'year',
                                        f'land_{year + out_len - 1}.tif')
        else:
            final_gt_tif = os.path.join(sim_dir,
                                        f'sim_{year + out_len - 1}.tif')
        prob2sim_main(final_gt_tif,
                      1,
                      prob_dir,
                      sim_dir,
                      demands[year_idx:year_idx+1],
                      os.path.join(data_dir, 'restriction.tif'),
                      os.path.join(data_dir, 'range.tif'),
                      True)
        year_idx += 1
        print(f'Simulate autoregressively, finished year {year}')


def test_year_autoreg(start_year: int,
                      first_simed_year:int,
                      first_year_tosim: int,
                      out_len: int,
                      data_dir: str,
                      model_path: str,
                      sim_dir: str,
                      prob_dir: str,
                      tile_size: int,
                      tile_step: int,
                      tile_start_idx: int = 0,
                      strategy: str = 'mean',
                      batch_size: int = 128,
                      num_workers: int = 0,
                      dataset_type: Literal['default', 'landscape'] = 'default'):
    """
    案例：使用2006-2011年作为输入，生成2012-2017的转换概率图
    Args:
        start_year:     输入数据的第一个年份，案例中为2006
        first_year_tosim: 模拟的第一个年份，案例中为2012
        out_len:        模拟的年份数目，案例中为2017-2012+1=6
        data_dir:       数据根目录
        model_path:     模型文件路径
        prob_dir:       输出结果保存路径
        batch_size:     批量大小
        num_workers:    线程数

    Returns:

    """

    assert first_year_tosim > 1900 and round(first_year_tosim) == first_year_tosim, \
        "Invalid first_year_tosim, required first_year_tosim>1900 and first_year_tosim integer"
    first_year_tosim = int(first_year_tosim)





    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.start_year = start_year
    args.in_len = first_year_tosim - start_year
    args.out_len = out_len
    args.data_root_dir = data_dir

    args.model_path = model_path
    args.prob_dir = prob_dir
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.tile_size = tile_size
    args.tile_step = tile_step
    args.tile_start_idx = tile_start_idx
    args.edge_width = 4

    args.range_tif = os.path.join(args.data_root_dir, 'range.tif')
    args.input_tifs = []
    for year in range(start_year, first_simed_year):
        args.input_tifs.append(os.path.join(args.data_root_dir, 'year', f'land_{year}.tif'))
    for year in range(first_simed_year, first_year_tosim):
        args.input_tifs.append(os.path.join(args.data_root_dir, sim_dir, f'sim_{year}.tif'))




    args.prob_tifs = [os.path.join(args.prob_dir, f'prob_{year}.tif') for year in
                      range(args.start_year + args.in_len, args.start_year + args.in_len + args.out_len)]
    args.use_mix = True
    args.autoreg = True
    args.dataset_type = dataset_type
    args.strategy = strategy
    test_model(args)



def test_model(args):
    saver = GDALImage(args.range_tif)
    range_arr = GDALImage.read_single(args.range_tif, -9999)

    model_info = torch.load(args.model_path)
    args.spa_var_tifs = model_info['spa_vars']
    if args.dataset_type == 'default':
        args.band = 1 + len(args.spa_var_tifs)
    elif args.dataset_type == 'landscape':
        args.band = 2 + len(args.spa_var_tifs)
    print(f'Model using spatial variable: {",".join(args.spa_var_tifs)}')
    print(f'Model filter size: {model_info["filter_size"]}')
    print(f'Model nlayers: {model_info["nlayers"]}')

    model = CC_ConvLSTM.CC_ConvLSTM(args.band, 1, 64,
                        model_info['filter_size'],
                        model_info['nlayers'],
                        64,
                        6, 6)
    model.in_len = args.in_len
    model.out_len = args.out_len
    model.load_state_dict(model_info['state_dict'])
    model.cuda()
    if args.dataset_type == 'default':
        dataset = CommonDataset(args.data_root_dir,
                                args.input_tifs,
                                args.spa_var_tifs,
                                False,
                                0,
                                args.tile_size,
                                args.tile_step,
                                args.tile_start_idx)
    elif args.dataset_type == 'landscape':
        raise NotImplementedError()
        # dataset = DatasetWithLandscape(
        #     [PatchSize()],
        #     args.data_root_dir,
        #     args.input_tifs,
        #     args.spa_var_tifs,
        #     False,
        #     0,
        #     args.tile_size,
        #     args.tile_step,
        #     args.tile_start_idx
        #     )
    else:
        raise NotImplementedError()

    tester = Tester(model, args, dataset,
                    range_arr, device, False, args.strategy)
    tester.loop()
    del dataset # 尽可能删除，非常占内存
    prob_arrs = tester.prob_arr
    for arr, tif_path in zip(prob_arrs, args.prob_tifs[:]):
        if not args.autoreg:
            arr[range_arr!=1] = -9999
        saver.save_block(arr, tif_path, gdal.GDT_Float32, no_data_value = -9999)
