import numpy as np
from .utils import *
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

def calc_fom(gt: np.ndarray, gn: np.ndarray, prev: np.ndarray):
    diff_count = ((gt - prev) == 1).sum()
    gn_diff = gn - prev
    gt_diff = gt - prev
    rc_sc = ((gn_diff * gt_diff) == 1).sum()
    fom = rc_sc/(2 * (diff_count - rc_sc) + rc_sc)

    return fom

def get_simulation_result(prev: np.ndarray, prob_map: np.ndarray, water: np.ndarray, land_demand: int) -> np.ndarray:
    sim = deepcopy(prev)
    prob_map[(prev == 1) | (water == 1)] = -9999
    indices = np.unravel_index(np.argsort(-prob_map, axis=None), prob_map.shape)
    indices = np.column_stack(indices)[:land_demand]

    sim[indices[:, 0], indices[:, 1]] = 1
    return sim

def set_random_arr(template:np.ndarray) -> np.ndarray:

    tshape = template.shape
    rd_arr = np.random.rand(tshape[0], tshape[1])
    e_arr  = -np.log(rd_arr)
    rd_arr  = np.power(e_arr, 0.5)

    rec_arr = template * rd_arr
    return rec_arr



def generate_simulation(prob_maps: list,
                        final_gt_map: np.ndarray,
                        water_map: np.ndarray,
                        range_map: np.ndarray,
                        img_saver: GDALImage,
                        out_tifs: list, 
                        land_demands: list) -> None:
    prev = final_gt_map # 2010 gt
    prev[range_map != 1] = -9999
    for prob_map, out_tif, land_demand in tqdm(zip(prob_maps, out_tifs, land_demands)):
        prob_map = set_random_arr(prob_map)
        prob_map[range_map != 1] = -9999
        cur_sim = get_simulation_result(prev, prob_map, water_map, land_demand)
        img_saver.save_block(cur_sim, out_tif, gdal.GDT_Int16, no_data_value=-9999)
        prev = cur_sim
        print(f'finish converting {out_tif}')


