from m2m_core.train import train_main

import argparse

parser0 = argparse.ArgumentParser()
parser0.add_argument('--start_year', default=2000, type=int)
parser0.add_argument('--in_len', default=6, type=int)
parser0.add_argument('--out_len', default=6, type=int)
parser0.add_argument('--data_dir', default=r'd://works2022//data-yrd', type=str)
parser0.add_argument('--spa_vars', type=str, nargs="+", default=['county.tif', 'town.tif', 'slope.tif'])
parser0.add_argument('--batch_size', default=8, type=int)
parser0.add_argument('--lr', default=0.00001, type=float)
parser0.add_argument('--sample_count', default=5000, type=int)
parser0.add_argument('--val_prop', default=0.15, type=float)
#
#
args0 = parser0.parse_args()
print(args0)
# args0.spa_vars = [eval(s) for s in args0.spa_vars]
train_main(args0.start_year,
           args0.in_len,
           args0.out_len,
           args0.data_dir,
           args0.spa_vars,
           args0.batch_size,
           args0.lr,
           args0.sample_count,
           args0.val_prop,
           'default'
           )
