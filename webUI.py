import json
import os, sys
import time
import glob
import gradio as gr
import subprocess
from gr_utils.utils import (tk_askfile_asy, tk_asksavefile_asy, tk_askdir_asy,
                            tk_askmultifile_asy, rangetif2img, get_datadir_info)
import platform

if platform.system() == 'Windows': # no problem in linux
    exe_path =os.path.dirname(sys.executable)
    os.environ['PROJ_LIB'] = os.path.join(exe_path,
                                          r'Lib\site-packages\osgeo\data\proj')


from m2m_core.FoM import calc_fom
from m2m_core.split import split_check_main
from m2m_core.normalize import normalize_main
from m2m_core.prob2sim import prob2sim_main
from m2m_core.test import test_main

from config_py import save_config, load_config, path_config

def gr_norm(input_tif, output_tif: str):
    try:
        out_dir = os.path.dirname(output_tif)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not output_tif.endswith('.tif'):
            return 'Invalid out file name, only .tif is supported'
        normalize_main(input_tif, output_tif)
        return f'Normalization finished, {output_tif}'
    except Exception as e:
        return e

def gr_calc_fom(final_observed_map, target_observed_map, target_simulated_map):
    if not os.path.exists(final_observed_map)\
        or not os.path.exists(target_simulated_map)\
        or not os.path.exists(target_simulated_map):
        gr.Warning('Invalid input tif')
        return

    try:
        print(final_observed_map, target_observed_map)
        fom = calc_fom(final_observed_map, target_observed_map, target_simulated_map)
        return str(fom*100) + '%'
    except Exception as e:
        gr.Warning(str(e))

def gr_split(data_dir,
             st_year,
             ed_year,
             progress = gr.Progress(track_tqdm=True)):
    if not os.path.exists(data_dir) or data_dir=='':
        gr.Warning('Invalid data dir')
        return

    var_dir = os.path.join(data_dir, 'vars')
    spa_vars = glob.glob(f'{var_dir}/*.tif')
    try:
        split_check_main(spa_vars,
                         data_dir,
                         int(st_year),
                         int(ed_year))
    except Exception as e:
        gr.Warning(str(e))
        return
    out_info = get_datadir_info(data_dir)
    range_img = rangetif2img(data_dir)
    return out_info, range_img


def gr_p2s(data_dir: str,
           final_gt_tif: str,
           years_to_sim: int,
           prob_dir: str,
           sim_dir: str,
           land_demands: str,
           progress = gr.Progress(track_tqdm=True)):
    if not os.path.exists(data_dir) or data_dir=='':
        gr.Warning('Invalid data dir')
        return

    try:
        land_demands = list(map(int, land_demands.split(',')))
    except Exception as e:
        info = f'{e}, invalid demands'
        return info
    restr_tif = os.path.join(data_dir, 'restriction.tif')
    range_tif = os.path.join(data_dir, 'range.tif')
    try:
        prob2sim_main(final_gt_tif,
                      years_to_sim,
                      prob_dir,
                      sim_dir,
                      land_demands,
                      restr_tif,
                      range_tif)
        return 'Conversion finished'
    except Exception as e:
        gr.Warning(str(e))

def gr_train(start_year: int,
             in_len: int,
             out_len: int,
             data_dir: str,
             spa_vars,
             batch_size: int,
             lr: float,
             sample_count: int,
             val_prop: float):
    print(sys.executable)
    if not os.path.exists(data_dir) or data_dir=='':
        gr.Warning("Invalid data dir")
        return
    try:
        spa_var_str = ''
        for spa_var in spa_vars.split('\n'):
            spa_var_str += f'{os.path.basename(spa_var)} '
        print(spa_var_str)
        cmd = f'python train_gui.py --start_year {int(start_year)} ' \
              f'--in_len {int(in_len)} ' \
              f'--out_len {int(out_len)} ' \
              f'--data_dir {data_dir} ' \
              f'--spa_vars {spa_var_str} ' \
              f'--batch_size {int(batch_size)} ' \
              f'--lr {lr} ' \
              f'--sample_count {int(sample_count)} ' \
              f'--val_prop {val_prop}'
        print(cmd)
        popen_cmd = (["cmd", "/c", "start", "cmd", "/k", cmd])
        subprocess.Popen(popen_cmd)
        return cmd
    except Exception as e:
        gr.Warning(str(e))

def gr_test(start_year: int,
            fs_year: int,
            out_len: int,
            data_dir: str,
            prob_dir: str,
            model_path,
            batch_size: int,
            num_workers: int,
            progress = gr.Progress(track_tqdm=True)):
    try:
        if not os.path.exists(data_dir) or data_dir=='':
            gr.Warning('Invalid data dir')
            return
        test_main(start_year,
                  fs_year,
                  out_len,
                  data_dir,
                  model_path.name,
                  prob_dir,
                  64, 48, 0,
                  batch_size,
                  num_workers)
        return 'finish testing'
    except Exception as e:
        gr.Warning(str(e))

async def gr_set_dir():
    dir_name = await tk_askdir_asy(init_dir=path_config.latest_path)
    path_config.latest_path = dir_name
    return dir_name

async def gr_set_tif():
    fname = await tk_askfile_asy(init_dir=path_config.latest_path, suffix='.tif')
    path_config.latest_path = os.path.basename(fname)
    if not fname.endswith('.tif') and fname!='':
        fname += '.tif'
    return fname

async def gr_set_outtif():
    fname = await tk_asksavefile_asy(init_dir=path_config.latest_path, suffix='.tif')
    path_config.latest_path = os.path.basename(fname)
    if not fname.endswith('.tif') and fname!='':
        fname += '.tif'
    return fname

async def gr_select_svars():
    svars = await tk_askmultifile_asy(init_dir=path_config.latest_path, suffix='.tif')
    new_svars = '\n'.join(svars)
    path_config.latest_path = os.path.basename(svars[0])
    return new_svars


with open('./config/miku.json', 'r') as mi:
    miku_json = json.load(mi)

theme = gr.Theme.from_dict(miku_json)
css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 16px !important}

#siderow {flex: 15%}
#maintabs {flex: 65%}
#father_row {display: flex; margin-top: 3%}

#test_year_col {flex: 10%}
#test_para_col {flex: 10%}
#test_file_col {flex: 40%}
#test_main_row {display: flex-inline}


#config_btn {margin-top:20%}
"""

with gr.Blocks(theme = theme, css = css) as demo:
    with gr.Column():
        with gr.Row(elem_id='father_row'):
            with gr.Column(elem_id='siderow'):
                dr_datadir_text = gr.Textbox(label='Data directory', elem_classes='feedback')
                dr_datadir_btn = gr.Button('Set data directory')
                dr_datadir_btn.click(gr_set_dir, inputs=[], outputs=[dr_datadir_text])
                with gr.Column(elem_id='config_btn'):
                    cfg_load_config_btn = gr.Button('Load config')
                    cfg_save_config_btn = gr.Button('Save config')
            with gr.Column(elem_id='maintabs'):
                with gr.Tabs():
                    with gr.TabItem("Normalization"):
                        with gr.Column():
                            with gr.Row():
                                norm_input_text = gr.Text(label='Input tif for normalization')
                                norm_input_btn = gr.Button('Select input tif')
                                norm_input_btn.click(gr_set_tif, inputs=[], outputs=[norm_input_text])
                            with gr.Row():
                                norm_output_text = gr.Text(label='output tif for normalization')
                                norm_output_btn = gr.Button('Select output tif')
                                norm_output_btn.click(gr_set_outtif, inputs=[], outputs=[norm_output_text])
                            with gr.Row():
                                norm_info_text = gr.Text(label='info')
                                norm_btn = gr.Button('Normalize')
                                norm_btn.click(gr_norm, [norm_input_text, norm_output_text], [norm_info_text])

                    with gr.TabItem("Checking raster"):
                        with gr.Row():
                            check_startyear_number = gr.Number(label='Start year', value=2006)
                            check_endyear_number = gr.Number(label='End year', value=2017)
                            check_btn = gr.Button('Check')
                        with gr.Row():
                            check_result_textarea   = gr.TextArea(label='Info')
                            check_range_image = gr.Image(label='range.tif')
                        check_btn.click(gr_split,
                                        inputs=[dr_datadir_text,
                                                   check_startyear_number,
                                                   check_endyear_number
                                                   ],
                                        outputs=[check_result_textarea, check_range_image])

                    with gr.TabItem("Train model"):
                        with gr.Blocks():
                            with gr.Row():
                                with gr.Column():
                                    tr_svars_textarea = gr.TextArea(label='Spatial variables', interactive=False)
                                    tr_select_svars_btn = gr.Button('Select spatial variables for training')
                                with gr.Blocks(title='Input data setting'):
                                    with gr.Column():
                                        tr_styear_number = gr.Number(label='First input year for training',
                                                                     minimum=1900, value=2000)
                                        tr_inlen_number = gr.Number(label='Count of years to input', minimum=0, value=6)
                                        tr_outlen_number = gr.Number(label='Count of years to simulate', minimum=0, value=6)
                                with gr.Blocks(title='Parameters'):
                                    with gr.Column():
                                        with gr.Row():
                                            tr_bsize_number = gr.Number(value=8, label='Batch size', minimum=0, maximum=1024)
                                            tr_lr_number = gr.Number(value=0.00001, label='Learning rate', minimum=0, maximum=1)
                                        with gr.Row():
                                            tr_scount_number = gr.Number(value=5000, label='Count of sampled tiles for training', minimum=100)
                                            tr_valprop_number = gr.Number(value=0.1, label='Validation proportion', minimum=0, maximum=1)
                                tr_select_svars_btn.click(gr_select_svars, outputs=[tr_svars_textarea])

                            tr_exe_button = gr.Button('Train model')
                            tr_exe_button.click(gr_train,
                                                inputs=[tr_styear_number, tr_inlen_number, tr_outlen_number,
                                                        dr_datadir_text, tr_svars_textarea, tr_bsize_number, tr_lr_number,
                                                        tr_scount_number, tr_valprop_number])

                    with gr.TabItem("Test model"):
                        with gr.Blocks():
                            with gr.Row(elem_id='test_main_row'):
                                with gr.Column(elem_id='test_year_col', min_width = 50):
                                    te_styear_number = gr.Number(label='First input year for testing', minimum=1900,
                                                                 value=2006)
                                    te_fsyear_number = gr.Number(label='First output year', minimum=1900, value=2012)
                                    te_outlen_number = gr.Number(label='Count of years to simulate', minimum=0, value=6)
                                with gr.Column(elem_id='test_file_col', min_width=50):
                                    with gr.Row():
                                        te_pdir_text = gr.Textbox(label='Dir for probability maps')
                                        te_pdir_btn  = gr.Button('Select dir for saving probability maps')
                                    te_pdir_btn.click(gr_set_dir, outputs=[te_pdir_text])
                                    te_model_file = gr.File(label='pth model file', file_types=['.pth'])
                                with gr.Column(elem_id='test_para_col', min_width=50):
                                    te_bsize_number = gr.Number(value=128, label='Batch size', minimum=0)
                                    te_nworker_number = gr.Number(value=0, label='num_worker for torch.loader', minimum=0)

                            with gr.Row():
                                with gr.Column():
                                    te_exe_button = gr.Button('Test model')
                                    te_out_text = gr.Text(label='info')
                                    te_exe_button.click(gr_test,
                                                        inputs=[te_styear_number, te_fsyear_number, te_outlen_number,
                                                        dr_datadir_text, te_pdir_text, te_model_file,
                                                        te_bsize_number, te_nworker_number],
                                                        outputs=te_out_text)

                    with gr.TabItem("Convert prob maps to urban land maps"):
                        with gr.Blocks():
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        p2s_gt_text = gr.Text(label='Final input land map')
                                        p2s_gt_btn  = gr.Button('Select final input land map')
                                        p2s_gt_btn.click(gr_set_tif, outputs=[p2s_gt_text])
                                    with gr.Row():
                                        p2s_pdir_text = gr.Textbox(label='Probability map dir')
                                        p2s_pdir_btn  = gr.Button('Select dir for probability maps')
                                        p2s_pdir_btn.click(gr_set_dir, outputs=[p2s_pdir_text])
                                    with gr.Row():
                                        p2s_sdir_text = gr.Textbox(label='Simulated map dir')
                                        p2s_sdir_btn  = gr.Button('Select dir for simulated maps')
                                        p2s_sdir_btn.click(gr_set_dir, outputs=[p2s_sdir_text])
                                with gr.Column():
                                    p2s_years_number = gr.Number(label='Count of years to be simulated', minimum=0,
                                                                 maximum=100,)
                                    p2s_demand_text = gr.Textbox(label='Land demands, seperated by comma')
                                    p2s_out_text    = gr.Textbox(label='Info')
                                    p2s_exe_button  = gr.Button('Convert')
                                p2s_exe_button.click(gr_p2s,
                                                     [dr_datadir_text,
                                                      p2s_gt_text,
                                                      p2s_years_number,
                                                      p2s_pdir_text,
                                                      p2s_sdir_text,
                                                      p2s_demand_text], p2s_out_text)

                with gr.Tabs():
                    with gr.TabItem("Calculate FoMs"):
                        with gr.Row():
                            gr_fom_sttif_text = gr.Text(label='Final input land map')
                            gr_fom_sttif_btn = gr.Button('Select final input land map')
                            gr_fom_sttif_btn.click(gr_set_tif, inputs=[], outputs=[gr_fom_sttif_text])
                        with gr.Row():
                            gr_fom_gttif_text = gr.Text(label='Observed land map of the target year')
                            gr_fom_gttif_btn = gr.Button('Select observed land map of the target year')
                            gr_fom_gttif_btn.click(gr_set_tif, inputs=[], outputs=[gr_fom_gttif_text])
                        with gr.Row():
                            gr_fom_simtif_text = gr.Text(label='Simulated land map of the target year')
                            gr_fom_simtif_btn = gr.Button('Select simulated land map of the target year')
                            gr_fom_simtif_btn.click(gr_set_tif, inputs=[], outputs=[gr_fom_simtif_text])
                        with gr.Row():
                            gr_fom_info_text = gr.Text(label='FoM')
                            gr_fom_calc_btn = gr.Button('Calculate')
                            gr_fom_calc_btn.click(gr_calc_fom,
                                                  [gr_fom_sttif_text, gr_fom_gttif_text, gr_fom_simtif_text],
                                                  [gr_fom_info_text])
                    with gr.TabItem("Predict land demands"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    gr.Text()
                                    gr.Button
                                with gr.Row():
                                    gr.Text()
                                    gr.Button
                            with gr.Column():
                                gr.Number()
                                gr.Checkboxgroup()
                            gr.Button
                            gr.TextArea()
        cfg_load_config_btn.click(load_config, [], [dr_datadir_text,
                                                    tr_svars_textarea,
                                                    tr_styear_number,
                                                    tr_inlen_number,
                                                    tr_outlen_number,
                                                    tr_bsize_number,
                                                    tr_lr_number,
                                                    tr_scount_number,
                                                    tr_valprop_number,
                                                    te_styear_number,
                                                    te_fsyear_number,
                                                    te_outlen_number,
                                                    te_pdir_text,
                                                    te_bsize_number,
                                                    te_nworker_number,
                                                    te_model_file,
                                                    p2s_gt_text,
                                                    p2s_pdir_text,
                                                    p2s_sdir_text,
                                                    p2s_demand_text
                                                    ])
        cfg_save_config_btn.click(save_config, [dr_datadir_text,
                                                tr_svars_textarea,
                                                tr_styear_number,
                                                tr_inlen_number,
                                                tr_outlen_number,
                                                tr_bsize_number,
                                                tr_lr_number,
                                                tr_scount_number,
                                                tr_valprop_number,
                                                te_styear_number,
                                                te_fsyear_number,
                                                te_outlen_number,
                                                te_pdir_text,
                                                te_bsize_number,
                                                te_nworker_number,
                                                te_model_file,
                                                p2s_gt_text,
                                                p2s_pdir_text,
                                                p2s_sdir_text,
                                                p2s_demand_text
                                                ])
    demo.queue(concurrency_count=1022, max_size=2044).launch(server_name="127.0.0.1",inbrowser=True,quiet=True,
                                                            share=False)

