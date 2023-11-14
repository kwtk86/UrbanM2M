import os

from easydict import EasyDict
from gr_utils.utils import tk_asksavefile_asy, tk_askfile_asy
import json
import gradio as gr
import os

path_config = EasyDict()
# path_config.latest_path = './'
# try:
#     with open('./config/dialog_cfg.json', 'r') as f:
#         j = json.load(f)
#         lpath = j['latest_path']
#         if os.path.exists(lpath):
#             path_config.latest_path = j['latest_path']
#         else:
#             path_config.latest_path = './'
# except:
path_config.latest_path = os.getcwd()


async def save_config(
                data_dir,
                train_spa_vars,
                train_first_year,
                train_input_years,
                train_output_years,
                train_batch_size,
                train_lr,
                train_samples,
                train_val_prop,
                test_first_ob_year,
                test_first_sim_year,
                test_output_years,
                test_prob_dir,
                test_batch_size,
                test_num_workers,
                test_pth,
                cvt_final_ob_tif,
                cvt_prob_dir,
                cvt_sim_dir,
                cvt_land_demands):
    config = EasyDict()

    config.datadir = data_dir

    config.train = EasyDict()
    config.train.svars = train_spa_vars
    config.train.first_year = train_first_year
    config.train.input_years = train_input_years
    config.train.output_years = train_output_years
    config.train.batch_size = train_batch_size
    config.train.lr = train_lr
    config.train.sample_tiles = train_samples
    config.train.val_prop = train_val_prop

    config.test = EasyDict()
    config.test.first_ob_year = test_first_ob_year
    config.test.first_sim_year = test_first_sim_year
    config.test.output_years = test_output_years
    config.test.prop_dir = test_prob_dir
    config.test.batch_size = test_batch_size
    config.test.num_workers = test_num_workers
    config.test.pth_file = test_pth

    config.convert = EasyDict()
    config.convert.final_ob_tif = cvt_final_ob_tif
    config.convert.prob_dir = cvt_prob_dir
    config.convert.sim_dir = cvt_sim_dir
    config.convert.land_demands = cvt_land_demands

    config.fom = EasyDict()

    path = await tk_asksavefile_asy('./config', '.json')
    if not path.endswith('.json'):
        path += '.json'
    with open(path, 'w') as f:
        di = dict(config)
        f.write(json.dumps(di))

# save_config('./config/cfg.json', config)
async def load_config():
    path = await tk_askfile_asy('config', '.json')
    with open(path, 'r') as f:
        di = json.load(f)
        config = EasyDict(di)
        
    try:
        return (config.datadir,
                config.train.svars,
                config.train.first_year,
                config.train.input_years,
                config.train.output_years,
                config.train.batch_size,
                config.train.lr,
                config.train.sample_tiles,
                config.train.val_prop,
                config.test.first_ob_year,
                config.test.first_sim_year,
                config.test.output_years,
                config.test.prop_dir,
                config.test.batch_size,
                config.test.num_workers,
                config.test.pth_file,
                config.convert.final_ob_tif,
                config.convert.prob_dir,
                config.convert.sim_dir,
                config.convert.land_demands
                )
    except Exception as e:
        gr.Error(f'Failed to load config\nError:{e}')


