# @Time : 2023/06/08 17:34
# @Author : 8
# @File : gdal_util.py
"""

"""
try:
    import gdal
except:
    from osgeo import gdal
import os
import numpy as np

class GDALImage:
    def __init__(self, template_img_name: str) -> None:
        tmp_dataset = gdal.Open(template_img_name)
        self.geo_trans = tmp_dataset.GetGeoTransform()
        self.prj = tmp_dataset.GetProjection()
        self.driver = gdal.GetDriverByName('GTiff')


    def save_multi_band_block(self, tif_array, tif_name, tif_type, no_data_value, geo_trans):
        band_count = tif_array.shape[0]
        dataset = self.driver.Create(tif_name,
                                     tif_array.shape[2],
                                     tif_array.shape[1],
                                     band_count,
                                     tif_type,
                                     options=["COMPRESS=LZW"])
        dataset.SetGeoTransform(geo_trans)
        dataset.SetProjection(self.prj)
        for i in range(1, band_count + 1):
            band = dataset.GetRasterBand(i)
            band.SetNoDataValue(no_data_value)
            band.WriteArray(tif_array[i - 1, :, :])
        dataset.FlushCache()



    def save_single_band_block(self, tif_array, tif_name, tif_type, no_data_value, geo_trans):
        dataset = self.driver.Create(tif_name,
                                     tif_array.shape[1],
                                     tif_array.shape[0],
                                     1,
                                     tif_type,
                                     options=["COMPRESS=LZW"])
        dataset.SetGeoTransform(geo_trans)
        dataset.SetProjection(self.prj)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(no_data_value)
        band.WriteArray(tif_array)
        dataset.FlushCache()


    def save_block(self,
                   tif_array: np.ndarray,
                   tif_name: str,
                   tif_type = gdal.GDT_Float32,
                   no_data_value: int = -9999,
                   start_row: int = None,
                   start_col: int = None):
        # if tif_type==gdal.GDT_Int16:
        #     tif_array[tif_array >= 2**16] = 2**16 - 1
        if start_row:
            geo_trans = (self.geo_trans[0] + start_col * self.geo_trans[1],
                         self.geo_trans[1],
                         self.geo_trans[2],
                         self.geo_trans[3] + start_row * self.geo_trans[5],
                         self.geo_trans[4],
                         self.geo_trans[5])
        else:
            geo_trans = self.geo_trans

        if len(tif_array.shape) == 3:
            self.save_multi_band_block(tif_array, tif_name, tif_type, no_data_value, geo_trans)
        elif len(tif_array.shape) == 2:
            self.save_single_band_block(tif_array, tif_name, tif_type, no_data_value, geo_trans)
        else:
            raise NotImplementedError()


    @staticmethod
    def read_single(ras: str, no_data_value: int or float) -> np.ndarray:
        dataset = gdal.Open(ras, gdal.GA_ReadOnly)
        band_count = dataset.RasterCount
        arrs = []
        # dateset_ndv = band.GetNoDataValue() #正常的ndv获取方法，但是这边要转换类型为int32
        for i in range(1, band_count + 1):
            band = dataset.GetRasterBand(i)
            arr = band.ReadAsArray().astype(np.float32)
            dataset_ndv = arr[0, 0]  # 特殊情况，仅使用于部分数据集
            arr[arr == dataset_ndv] = no_data_value  # 统一不同数据集的ndv
            arrs.append(arr)
        if len(arrs)>1:
            tot_arr = np.stack(arrs)
        else:
            tot_arr = arrs[0]
        return tot_arr
