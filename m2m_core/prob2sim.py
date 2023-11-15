from .config import *
import os, sys
from .utils.future_land import *
import glob

def prob2sim_main(final_gt_tif: str,
                  years_to_sim: int,
                  prob_dir: str,
                  sim_dir: str,
                  land_demands: list,
                  restriction_tif: str,
                  range_tif: str,
                  autoreg: bool = False):
    """

    Args:
        autoreg:
        years_to_sim:
        final_gt_tif: 最后一年的城市用地二值栅格路径（真值） gt的意思是ground truth
        prob_dir:     测试阶段获得的转换概率栅格保存的文件夹
        sim_dir:      生成的模拟结果保存文件夹
        land_demands: 用地需求量列表
        restriction_tif: 限制发展栅格
        range_tif:       范围栅格

    Returns:

    """
    try:
        first_prob_year = int(os.path.basename(final_gt_tif)[:-4].split('_')[-1]) + 1
    except:
        raise RuntimeError('Invalid final gt tif')
    try:
        prob_tifs = []
        for year in range(first_prob_year, first_prob_year + years_to_sim):
            prob_tifs.append(f'{prob_dir}/prob_{year}.tif')
        # prob_tifs = glob.glob(f'{prob_dir}/prob*.tif')
        # prob_tifs.sort(key=lambda x:int(x.split('.tif')[0].split('_')[-1]))
    except:
        raise RuntimeError(f'Invalid prob tifs in {prob_dir}')

    assert os.path.exists(range_tif), f"{range_tif} does not exist"

    saver = GDALImage(range_tif)
    prob_arrs = [saver.read_single(tif_path, -9999) for tif_path in prob_tifs]
    sim_tifs = [os.path.join(sim_dir, os.path.split(tif)[-1].replace('prob', 'sim')) for tif in prob_tifs]

    assert len(land_demands) == len(prob_tifs), "Count of land demands must be equal to the years to be predicted"
    assert all([isinstance(de, int) for de in land_demands]), "Invalid land demands"

    range_arr = GDALImage.read_single(range_tif, -9999)
    if os.path.exists(restriction_tif):
        restriction_arr = GDALImage.read_single(restriction_tif, -9999)
    else:
        restriction_arr = np.zeros_like(range_arr)
        print('Failed to find restriction.tif, zero-filled array created')

    final_gt_arr = GDALImage.read_single(final_gt_tif, -9999)
    print('finish reading')

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if autoreg:
        generate_simulation_autoreg(prob_arrs,
                                    final_gt_arr,
                                    restriction_arr,
                                    range_arr,
                                    saver,
                                    sim_tifs,
                                    land_demands)
    else:
        generate_simulation(prob_arrs,
                            final_gt_arr,
                            restriction_arr,
                            range_arr,
                            saver,
                            sim_tifs,
                            land_demands)
