# @Time : 2023/12/11 16:33
# @Author : 8
# @File : gms_interface.py
"""
GMS 接口，将多个流程包装为一个流程
"""
import os, glob
from .test import test_main
from .normalize import normalize_folder
from .prob2sim import prob2sim_main
from .split import split_check_main


def gms_m2m_main(data_dir: str,
                 st_year: int,
                 first_sim_year: int,
                 out_len: int,
                 model_path: str,
                 prob_dir: str,
                 sim_dir: str,
                 land_demands: list,
                 batch_size: int,
                 num_workers: int) -> None:
    """
    GMS平台使用的M2M主程序接口
    以使用2006-2011的数据模拟2012-2017的变化情况进行说明
    Args:
        data_dir:     数据文件夹
        st_year:      起始年份，由用户输入。案例中为为2006
        first_sim_year: 第一个模拟年份，由用户输入。案例中为2012
        out_len:      模拟输出的年份数量，由用户输入。案例中为6（2017-2012+1=6）
        model_path:   模型权重文件路径
        prob_dir:     概率图保存路径，建议放在datadir下
        sim_dir:      模拟结果保存路径，建议放在datadir下
        land_demands: 用地需求量，由用户输入。列表长度应=outlen
        batch_size:   建议根据GMS服务器性能确定
        num_workers:  建议根据GMS服务器性能确定

    Returns:

    """
    in_var_dir = os.path.join(data_dir, 'vars0')
    norm_var_dir = os.path.join(data_dir, 'vars')
    normalize_folder(in_var_dir, norm_var_dir)
    split_check_main(glob.glob(f'{norm_var_dir}/*.tif'),
                     data_dir, st_year, first_sim_year+out_len-1)
    test_main(st_year, first_sim_year, out_len,
              data_dir, model_path, prob_dir,
              64, 48, 0,
              batch_size, num_workers, 'mean')
    final_gt_tif = os.path.join(data_dir, 'year', f'land_{first_sim_year-1}.tif')
    restr_tif = os.path.join(data_dir, 'restriction.tif')
    range_tif = os.path.join(data_dir, 'range.tif')
    prob2sim_main(final_gt_tif, out_len,
                  prob_dir, sim_dir, land_demands,
                  restr_tif, range_tif, False)


from .FoM import calc_fom

def fom_validation(start_tif: str, ground_truth_tif: str, simulated_tif: str) -> float:
    """
    OpenGMS平台进行FoM验证的代码接口
    比如运行模型时，使用2006-2011的数据模拟了2012-2017的城市扩张
    则fom基于 2011年真实用地数据（最后一个已知年份） 2017年真实用地数据 及 2017年模拟结果（目标年份）进行计算

    基本原理是：基于 2011年真实用地数据 2017年真实用地数据，得到真实城市扩张情况
               再基于2011年真实用地数据 2017年模拟结果进行计算 得到模拟的城市扩张情况
               对比真实与模拟，计算FOM

    Args:
        start_tif:        最后一个已知年份的真实用地栅格
        ground_truth_tif: 目标年份的真实用地栅格
        simulated_tif:    目标年份的用地模拟结果

    Returns:
        计算结果
    """
    fom = calc_fom(start_tif, ground_truth_tif, simulated_tif)
    return fom