from m2m_core.test import test_autoreg_main
import os, sys
import platform

if platform.system() == 'Windows': # no problem in linux
    exe_path =os.path.dirname(sys.executable)
    os.environ['PROJ_LIB'] = os.path.join(exe_path,
                                          r'Lib\site-packages\osgeo\data\proj')
demands = [
584425,
661210,
431138,
491400,
449350,
790081,
]
for epoch in range(20, 28):
    test_autoreg_main(2006,
                      2012,
                      6,
                      r'D:\works2022\data-yrd',
                      fr'E:\15-4.gradio/trained_models/convlstm-t08_14_23-e{epoch}.pth',
                      fr'D:\works2022\data-yrd\sim_e{epoch}',
                      r'D:\works2022\data-yrd\prob',
                      demands,
                      tile_step=48,
                      strategy='max',
                      dataset_type='default')

