import os
import glob
from osgeo import gdal, osr
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename, askopenfilenames
import asyncio
import numpy as np
from PIL import Image

def tk_window_askfile(init_dir=os.getcwd(), suffix = '') -> str:
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    filename = askopenfilename(initialdir=init_dir,
                               filetypes=[('', suffix)])
    return filename

def tk_window_askmultifile(init_dir=os.getcwd(), suffix = '') -> str or list:
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    filename = askopenfilenames(initialdir=init_dir,
                               filetypes=[('', suffix)])
    return filename


def tk_window_asksavefile(init_dir=os.getcwd(), suffix = '') -> str:
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    filename = asksaveasfilename(initialdir=init_dir,
                               filetypes=[('', suffix)])
    return filename


def tk_window_askdir(init_dir=os.getcwd()) -> str:
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    filename = askdirectory(initialdir=init_dir)
    return filename


async def tk_askdir_asy(init_dir = os.getcwd()) -> str:
    fname = await asyncio.to_thread(tk_window_askdir, init_dir)
    return fname

async def tk_askfile_asy(init_dir = os.getcwd(), suffix = '') -> str:
    fname = await asyncio.to_thread(tk_window_askfile, init_dir, suffix)
    return fname

async def tk_askmultifile_asy(init_dir = os.getcwd(), suffix = '') -> str or list:
    fname = await asyncio.to_thread(tk_window_askmultifile, init_dir, suffix)
    return fname

async def tk_asksavefile_asy(init_dir = os.getcwd(), suffix = '') -> str:
    fname = await asyncio.to_thread(tk_window_asksavefile, init_dir, suffix)
    return fname


def get_datadir_info(data_dir: str) -> str:
    out_info = "Valid data directory for simulation!\nData directory information:\n-------------------\n"
    if os.path.exists(os.path.join(data_dir, 'restriction.tif')):
        out_info += "restriction.tif: exists\n-------------------\n"
    else:
        out_info += "restriction.tif: not exists or invalid\n-------------------\n"
    spa_vars = glob.glob(f'{os.path.join(data_dir, "vars")}/*.tif')
    spa_vars = [os.path.basename(tif) for tif in spa_vars]
    spa_vars = '\n\t'.join(spa_vars)
    out_info += f"Spatial variables: \n\t{spa_vars}\n-------------------\n"
    range_tif_info = gdal.Open(f'{data_dir}/range.tif')

    size_x, size_y = range_tif_info.RasterXSize, range_tif_info.RasterYSize
    out_info += f"Raster size:       {size_x}, {size_y}\n-------------------\n"

    geo_trans = range_tif_info.GetGeoTransform()
    res_x, res_y = geo_trans[1], abs(geo_trans[5])
    out_info += f"Raster resolution: {res_x}, {res_y}\n-------------------\n"

    prj = range_tif_info.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    out_info += f"Spatial reference: {srs.GetName()}\n"
    return out_info

def rangetif2img(data_dir: str) -> np.ndarray:
    tif_img = Image.open(os.path.join(data_dir, 'range.tif'))
    img_size = tif_img.size
    new_size = (800, int(800*(img_size[1]/img_size[0])))
    tif_img = tif_img.resize(new_size, Image.Resampling.NEAREST)
    tif_arr = np.array(tif_img)
    out_img = np.where(tif_arr==tif_arr[0,0], 255, 25).astype(np.int16)
    # out_img = np.array(Image.fromarray(out_img).resize(new_size, Image.Resampling.BILINEAR))
    out_img = np.stack((out_img, out_img, out_img), axis=2)
    return out_img
