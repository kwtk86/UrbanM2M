from .config import *

import argparse
import os, sys
from os.path import join as pj
from .utils.data_loading import Dataset_woSplit
from .utils.future_land import *
from .utils.trainers.tester import *
from .model import CC_ConvLSTM

def check_args(args):

    # args.tile_dir = os.path.join(args.data_root_dir, f'tile_test')
    args.range_tif = os.path.join(args.data_root_dir, 'range.tif')

    
    args.input_tifs = [f'land_{year}.tif' for year in range(args.start_year, args.start_year + args.in_len, 1)]

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
              batch_size: int,
              num_workers: int):
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
        "Invalid fisrt_sim_year, required fisrt_sim_year>1900 and fisrt_sim_year integer"
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

    # args.height = 64
    # args.block_step = 48
    args.edge_width = 4

    args = check_args(args)
    test_model(args)

def test_model(args):
    saver = GDALImage(args.range_tif)
    range_arr = GDALImage.read_single(args.range_tif, -9999)

    model_info = torch.load(args.model_path)
    args.spa_var_tifs = model_info['spa_vars']
    args.band = 1 + len(args.spa_var_tifs)
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
    dataset = Dataset_woSplit(args.data_root_dir,
                             args.input_tifs,
                             args.spa_var_tifs,
                             False,
                             0,
                             args.tile_size,
                             args.tile_step)
    tester = Tester(model, args, dataset, range_arr, device, True)
    tester.loop()
    del dataset # 尽可能删除，非常占内存
    prob_arrs = tester.prob_arr[args.in_len - 1:]
    for arr, tif_path in zip(prob_arrs, args.prob_tifs[:]):
        arr[range_arr!=1] = -9999
        saver.save_block(arr, tif_path, gdal.GDT_Float32, no_data_value = -9999)
