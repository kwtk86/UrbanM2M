import os
import torch
from torch.optim import lr_scheduler
import numpy as np
import glob
from osgeo import gdal


def get_scheduler(optimizer, args, t_max):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # if args.lr_policy == 'linear':
    #     niter = 5
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + args.epoch_count - niter) / float(args.niter_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # elif args.lr_policy == 'step':
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    # elif args.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    # elif args.lr_policy == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    # else:
    #     return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler



class GDALImage:
    def __init__(self, template_img_name:str) -> None:
        tmp_dataset = gdal.Open(template_img_name)
        self.geo_trans = tmp_dataset.GetGeoTransform()
        self.prj = tmp_dataset.GetProjection()

    def save_block(self,
                   tif_array:np.ndarray,
                   tif_name:str,
                   tif_type = gdal.GDT_Float32,
                   start_row: int = None,
                   start_col: int = None,
                   no_data_value: int = -9999) -> None:
        driver = gdal.GetDriverByName('GTiff')

        tif_dir = os.path.split(tif_name)[0]
        if not os.path.exists(tif_dir):
            os.makedirs(tif_dir, exist_ok=True)

        dataset = driver.Create(tif_name, tif_array.shape[1], tif_array.shape[0], 1, tif_type,
                                options=["COMPRESS=LZW"])
        if start_row:
            geo_trans = (self.geo_trans[0] + start_col * self.geo_trans[1],
                         self.geo_trans[1],
                         self.geo_trans[2],
                         self.geo_trans[3] + start_row *self.geo_trans[5],
                         self.geo_trans[4],
                         self.geo_trans[5])
            dataset.SetGeoTransform(geo_trans)
        else:
            dataset.SetGeoTransform(self.geo_trans)
        dataset.SetProjection(self.prj)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(tif_array)
        dataset.FlushCache()

    @staticmethod
    def read_single(ras: str, no_data_value: float, return_ndv: bool = False):
        dataset = gdal.Open(ras, gdal.GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        dataset_ndv = band.GetNoDataValue()
        arr = band.ReadAsArray()
        arr = np.where(arr == dataset_ndv, no_data_value, arr).astype(np.float32)
        if return_ndv:
            return arr, no_data_value
        else:
            return arr
    


def fuzzy_search_pth_path(pth_path0: str)-> str:
    # pths = glob.glob(os.path.split(pth_path0)[0] + '/*')
    fuzzy_search = glob.glob(f'{pth_path0}--*')
    if len(fuzzy_search) == 1:
        return fuzzy_search[0]
    else:
        print(pth_path0)
        print(os.listdir(os.path.split(pth_path0)[0])[0])
        raise RuntimeError('failed to find model')