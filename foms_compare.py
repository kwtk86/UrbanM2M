import os
from m2m_core.FoM import calc_fom

st = r'D:\works2022\data-yrd\year\land_2011.tif'
gt = r'D:\works2022\data-yrd\year\land_2017.tif'

foms1 = []
foms2 = []
for epoch in range(20, 28):
    sim1 = fr'D:\works2022\data-yrd\sim_e{epoch}\sim_2017_range.tif'
    foms1.append(calc_fom(st, gt, sim1))

    sim2 = fr'D:\works2022\data-yrd\sim2_e{epoch}\sim_2017.tif'
    foms2.append(calc_fom(st, gt, sim2))
print(foms1, foms2)
print(sum(foms1)/8, sum(foms2)/8)