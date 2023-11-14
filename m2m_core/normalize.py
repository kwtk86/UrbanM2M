import glob

from .utils.utils import GDALImage
import os
import numpy as np

def normalize_main(in_tif: str, out_tif: str):
    """

    Args:
        in_tif:
        out_tif:

    Returns:

    """
    if not os.path.exists(in_tif):
        raise RuntimeError('Input tif does not exist')

    out_dir = os.path.dirname(out_tif)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_arr = GDALImage.read_single(in_tif, -9999)
    valid_arr = in_arr[in_arr!=-9999]
    vmax, vmin = valid_arr.max(), valid_arr.min()
    out_arr = np.where(in_arr != -9999, (in_arr-vmin)/(vmax - vmin), in_arr)
    saver = GDALImage(in_tif)
    saver.save_block(out_arr, out_tif)


def normalize_folder(folder: str, out_folder: str):
    tifs = glob.glob(f'{folder}/*.tif')
    for tif in tifs:
        base_name = os.path.basename(tif)
        out_name = os.path.join(out_folder, base_name)
        normalize_main(tif, out_name)
    return "success"