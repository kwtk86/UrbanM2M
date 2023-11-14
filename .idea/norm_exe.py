import tkinter as tk
from tkinter import filedialog, messagebox

import os
import numpy as np
from osgeo import gdal

in_tif = ''
out_tif = ''

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
        arr = np.where(arr == dataset_ndv, no_data_value, arr)
        # arr = band.ReadAsArray().astype(np.float32)

        if return_ndv:
            return arr, dataset_ndv
        else:
            return arr

def normalize_main():
    """

    Args:
        in_tif:
        out_tif:

    Returns:

    """
    if in_tif == '':
        messagebox.showerror('Invalid input', 'Invalid input')
        return

    if out_tif == '':
        messagebox.showerror('Invalid output', 'Invalid output')
        return

    if not os.path.exists(in_tif):
        messagebox.showerror('Invalid input', 'Input tif does not exist')
        return

    out_dir = os.path.dirname(out_tif)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    try:
        in_arr = GDALImage.read_single(in_tif, -9999)
        valid_arr = in_arr[in_arr!=-9999]
        vmax, vmin = valid_arr.max(), valid_arr.min()
        out_arr = np.where(in_arr != -9999, (in_arr-vmin)/(vmax - vmin), in_arr)
        saver = GDALImage(in_tif)
        saver.save_block(out_arr, out_tif)
    except Exception as e:
        messagebox.showerror('Runtime error', e)
    messagebox.showinfo('Finish', 'finish!')

def choose_tif_file():
    global in_tif

    tif_file_path = filedialog.askopenfilename(filetypes=[("TIFF Files", "*.tif")])
    if tif_file_path:
        tif_label.config(text="选择的TIFF文件: " + tif_file_path)
        in_tif = tif_file_path

def choose_save_path():
    global out_tif

    save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Files", "*.tif")])
    if save_path:
        if not save_path.lower().endswith(".tif"):
            save_path += ".tif"
        save_label.config(text="保存路径: " + save_path)
        out_tif = save_path

# 创建主窗口
root = tk.Tk()
root.title("UrbanM2M Normalization")

# 设置界面样式
root.configure(bg="#f0f0f0")  # 设置背景颜色

# 创建容器框架
frame1 = tk.Frame(root, bg="#f0f0f0")
frame2 = tk.Frame(root, bg="#f0f0f0")
frame3 = tk.Frame(root, bg="#f0f0f0")

# 将容器框架打包
frame1.pack(fill=tk.X, padx=20, pady=(20, 10))
frame2.pack(fill=tk.X, padx=20, pady=10)
frame3.pack(fill=tk.X, padx=20, pady=10)

# 第一行：选择现存的tif文件
tif_button = tk.Button(frame1, text="Choose input TIFF", command=choose_tif_file)
tif_label = tk.Label(frame1, text="", bg="#f0f0f0")

tif_button.pack(side=tk.LEFT)
tif_label.pack(side=tk.LEFT, anchor="w")

# 第二行：选择tif文件保存路径
save_button = tk.Button(frame2, text="Choose save path", command=choose_save_path)
save_label = tk.Label(frame2, text="", bg="#f0f0f0")

save_button.pack(side=tk.LEFT)
save_label.pack(side=tk.LEFT, anchor="w")

# 第三行：调用norm函数
norm_button = tk.Button(frame3, text="Normalize", command=normalize_main, )
norm_button.pack(side=tk.LEFT)

# 启动主循环
root.mainloop()
