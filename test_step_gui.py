from m2m_core.test import test_autoreg_main
import os, sys
import platform

if platform.system() == 'Windows': # no problem in linux
    exe_path =os.path.dirname(sys.executable)
    os.environ['PROJ_LIB'] = os.path.join(exe_path,
                                          r'Lib\site-packages\osgeo\data\proj')
demands = [1200000, 1200000, 1200000, 1200000, 1200000, 1200000]

test_autoreg_main(2006,
                  2012,
                  6,
                  r'D://works2022/data-yrd-4km',
                  r'E:\15-4.gradio/trained_models/convlstm-t08_14_23-e15.pth',
                  r'D://works2022/data-yrd-4km/sim_stepr1',
                  r'D://works2022/data-yrd-4km/prob_stepr1',
                  demands,
                  tile_step=48,
                  strategy='max')



# demands = [300000]*30
# ssyear = 2006
# seyear = 2030
# for start_year in range(ssyear, seyear):
#     # if start_year < 2012:
#     #     continue
#     # else:
#     test_year_autoreg(start_year, 2012, start_year + 6, 1,
#                       'D://works2022/data-yrd-4km',
#               r'E:\15-4.gradio/trained_models/convlstm-t08_14_23-e15.pth',
#               'D://works2022/data-yrd-4km/sim_step_30',
#               'D://works2022/data-yrd-4km/prob_step_30', 64, 48, 32 * (idx//2),
#                       100, 0)
#     if start_year == ssyear:
#         final_gt_tif = f'D://works2022/data-yrd-4km/year/land_{start_year+5}.tif'
#     else:
#         final_gt_tif = f'D://works2022/data-yrd-4km/sim_step_30/sim_{start_year + 5}.tif'
#
#     prob2sim_main(final_gt_tif,
#                   1,
#                   'D://works2022/data-yrd-4km/prob_step_30',
#                   'D://works2022/data-yrd-4km/sim_step_30',
#                   demands[idx:idx+1],
#                   'D://works2022/data-yrd-4km/restriction.tif',
#                   'D://works2022/data-yrd-4km/range.tif',
#                   True)
#     print(start_year)