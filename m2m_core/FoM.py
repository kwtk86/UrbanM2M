import numpy as np
from .utils.utils import GDALImage

def calc_fom(st_path: str, gt_path: str, gn_path: str) -> float:
    
    gn, gn_ndv = GDALImage.read_single(gn_path, -9999, True)
    gt = GDALImage.read_single(gt_path, -9999)
    st = GDALImage.read_single(st_path, -9999)

    gt = np.where(gn==gn_ndv, gn_ndv, gt)
    st = np.where(gn==gn_ndv, gn_ndv, st)

    diff_count = ((gt - st) == 1).sum()
    gn_diff = gn - st
    gt_diff = gt - st
    rc_sc = ((gn_diff * gt_diff) == 1).sum()
    fom = rc_sc/(2 * (diff_count - rc_sc) + rc_sc)
    return fom