
import os
from .test import test_main
from .prob2sim import prob2sim_main
from .normalize import normalize_main
from .split import split_main


__all__ = ['test_convert', 'norm_split']




def norm_split(data_dir: str,
               st_year: int,
               ed_year: int):
    """
    归一化+生成切片
    案例：使用2006-2011年作为输入，生成2012-2017的转换概率图

    Args:
        data_dir: 数据根目录
        st_year: 切片起始年份。案例中为2006。
        ed_year: 切片终止年份。案例中为2017。

    Returns:

    """
    spa_vars = ['county.tif', 'slope.tif', 'town.tif']
    in_var_dir = os.path.join(data_dir, 'svars')
    out_var_dir = os.path.join(data_dir, 'vars')

    input_spavars = [os.path.join(in_var_dir, spa_var) for spa_var in spa_vars]
    normed_spavars = [os.path.join(out_var_dir, spa_var) for spa_var in spa_vars]

    for v in input_spavars:
        if not os.path.exists(v):
            return RuntimeError(f'{v} does not exist')

    if not os.path.exists(out_var_dir):
        os.makedirs(out_var_dir)

    try:
        for in_, out_ in zip(input_spavars, normed_spavars):
            normalize_main(in_, out_)
        split_main(normed_spavars, data_dir, st_year, ed_year, False, False, 64, 48)
    except Exception as e:
        raise RuntimeError(e)

def test_convert(data_dir: str,
              st_year: int,
              first_sim_year: int,
              out_len: int,
              model_path: str,
              prob_dir: str,
              sim_dir: str,
              land_demands: list,
              batch_size: int,
              num_workers: int):
    """
    测试，并将结果转为概率图
    案例：使用2006-2011年作为输入，生成2012-2017的转换概率图

    Args:
        data_dir:     
        st_year:      输入数据的第一个年份，案例中为2006
        first_sim_year: 模拟的第一个年份，案例中为2012
        out_len:      模拟的年份数目，案例中为2017-2012+1=6
        model_path:   模型文件路径
        prob_dir:     概率图保存路径
        sim_dir:      模拟结果保存路径
        land_demands: 用地需求量，列表
        batch_size:   批量大小
        num_workers:  线程数

    Returns:

    """
    try:
        test_main(st_year, first_sim_year, out_len, data_dir, model_path, prob_dir,batch_size, num_workers)
        restr_tif = os.path.join(data_dir, 'restriction.tif')
        range_tif = os.path.join(data_dir, 'range.tif')
        final_gt_tif = os.path.join(data_dir, 'year', f'land_{first_sim_year-1}.tif')
        prob2sim_main(final_gt_tif, prob_dir, sim_dir, land_demands, restr_tif, range_tif)
    except Exception as e:
        raise RuntimeError(e)
