import numpy as np
import torch
import os
import pylandstats as pls
from scipy import ndimage


class PLSBase:
    def __init__(self):
        pass

class PatchSize(PLSBase):
    def __init__(self, max_patch_size: int = 10000):
        super().__init__()
        self.max_patch_size = max_patch_size

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        arr2 = arr.copy()
        arr2[arr2 != 1] = 0
        labeled_array, num_features = ndimage.label(arr2)
        sizes = ndimage.sum(arr2, labeled_array, range(num_features + 1))
        arr3 = sizes[labeled_array]
        mmax = 1024
        arr3[arr3>mmax] = mmax
        arr3/=mmax
        return arr3

class LandscapeConfig:
    def __init__(self,
                 funcs: list):
        self.scape_funcs = funcs
        self.types = len(funcs)

    def get_landscape(self, arr: np.ndarray or torch.Tensor) -> torch.Tensor:
        res = []
        for func in self.scape_funcs:
            t_res = func(arr)
            res.append(t_res)
        return torch.Tensor(np.array(res))