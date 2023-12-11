from m2m_core.test import test_main
from m2m_core.prob2sim import prob2sim_main
import os, sys

os.environ['PROJ_LIB'] = os.path.join(os.path.dirname(sys.executable), r'Lib\site-packages\osgeo\data\proj')

demands = [
584425,
661210,
431138,
491400,
449350,
790081,
]

for epoch in range(20, 28):
    test_main(2006,
              2012,
              6,
               r'D:\works2022\data-yrd',
              fr'E:\15-4.gradio/trained_models/convlstm-t08_14_23-e{epoch}.pth',
               r'D:\works2022\data-yrd\prob2',
              64, 48, 0,
              100,
              0,
              'max',
              'default')
    prob2sim_main(r'D:\works2022\data-yrd\year\land_2011.tif',
                6,
                r'D:\works2022\data-yrd\prob2',
                fr'D:\works2022\data-yrd\sim2_e{epoch}',
                  demands,
                  r'D:\works2022\data-yrd\restriction.tif',
                  r'D:\works2022\data-yrd\range.tif',
                  False)