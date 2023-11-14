from typing import List
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from ..utils import *

from osgeo import gdal



class Tester:
    def __init__(self, model, args, dataset, template_arr, device, cal_loss):
        self.test_loader = None
        self.model     = model
        self.args      = args
        self.device    = device
        self.cal_loss  = cal_loss

        self.criterion = nn.MSELoss(size_average=False, reduce=True).to(self.device)
        self.in_len   = args.in_len
        self.out_len  = args.out_len

        self.dataset = dataset
        self.batch_size = args.batch_size
        self.block_size = args.tile_size
        self.block_step = args.tile_step
        self.edge_width = args.edge_width

        self.step_diff  = self.batch_size - self.block_step
        self.template_arr = template_arr
        self.region_x   = template_arr.shape[1]
        self.region_y   = template_arr.shape[0]

        self.prob_arr    = np.zeros((self.out_len + self.in_len - 1, self.region_y, self.region_x))
        self.overlap_arr = np.zeros((self.out_len + self.in_len - 1, self.region_y, self.region_x))

        self.get_loader()

    def get_loader(self):
        nw = self.args.num_workers
        if nw==0:
            self.test_loader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)
        else:
            self.test_loader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, num_workers=nw)

    def loop(self):
        self.test_loop(self.test_loader)

    def merge_arr(self, data: np.ndarray, sx: int, sy: int):
        sly = slice(sy + self.edge_width, sy + self.block_size - self.edge_width)
        slx = slice(sx + self.edge_width, sx + self.block_size - self.edge_width)
        self.prob_arr[:, sly, slx] += data[:, self.edge_width:-self.edge_width,
                                      self.edge_width:-self.edge_width]
        self.overlap_arr[:, sly, slx] += 1

    def post_process(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.prob_arr /= self.overlap_arr
        mask = (self.template_arr == -9999)
        self.prob_arr[np.isnan(self.prob_arr)] = 0
        self.prob_arr[:, mask] = -9999



    def test_loop(self, loader: DataLoader):
        tot_loss = 0
        with tqdm(total=len(loader), ncols = 75) as pbar:
            with torch.no_grad():
                for rc, spa_vars, x in loader:
                    loss, data = self.model_forward(x, spa_vars)

                    for b in range(spa_vars.shape[0]):
                        yxrc_lst = rc[b].split('_')
                        sy, sx = list(map(int, yxrc_lst[:2]))
                        sy, sx = sy - self.block_size, sx - self.block_size
                        self.merge_arr(data[b], sx, sy)
                    pbar.update(1)
                    pbar.set_postfix({'loss': float(loss), 'avg_loss': tot_loss / pbar.n})
                    tot_loss += float(loss)
        self.post_process()

    def model_forward(self, x: torch.Tensor, spa_vars: torch.Tensor, mask: list = []):
        spa_vars = spa_vars.to(self.device)
        x = x.to(self.device)
        mask = self.get_teacher_masker(x)
        gn_imgs = self.model(x, spa_vars, mask, True)
        gt_imgs = x[:, 1:]
        loss = self.criterion(gn_imgs[:, :self.in_len-1], gt_imgs)
        return loss, gn_imgs[:, :, 0].cpu().detach().numpy()

    def get_teacher_masker(self, x: torch.Tensor) -> List[torch.Tensor]:
        masks = [torch.FloatTensor(x.shape[0],
                                   x[0].size(1),
                                   x[0].size(2),
                                   x[0].size(3)).fill_(0).to(self.device)
                                   for i in range(self.out_len - 1)]
        return masks

