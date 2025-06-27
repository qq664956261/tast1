import inspect
import logging
import cv2
import torch
import importlib
import numpy as np
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import utils
# import models
from models.SuperPoint import SuperPointNet
from models.ALike import ALNet

# import tasks
from tasks.repeatability import repeatability, plot_repeatability



class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.matcher = None
        # model choice
        if params['model_type'] == 'SuperPoint':
            self.model = SuperPointNet()
            self.model.load_state_dict(torch.load(params['SuperPoint_params']['weight']))
        elif params['model_type'] == 'Alike':
            self.model = ALNet(params['Alike_params'])
            self.model.load_state_dict(torch.load(params['Alike_params']['weight']))
        else:
            raise NotImplementedError
        self.model.eval()
        self.num_feat = None
        self.repeatability = None
        self.rep_mean_err = None
        self.accuracy = None
        self.matching_score = None
        self.track_error = None
        self.last_batch = None
        self.fundamental_error = None
        self.fundamental_radio = None
        self.fundamental_num = None
        self.r_est = None
        self.t_est = None

    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.rep_mean_err = []
        self.accuracy = []
        self.matching_score = []
        self.track_error = []
        self.fundamental_error = []
        self.fundamental_radio = []
        self.fundamental_num = []
        self.r_est = [np.eye(3)]
        self.t_est = [np.zeros([3, 1])]

    def on_test_end(self) -> None:
        self.num_feat = np.mean(self.num_feat)
        self.accuracy = np.mean(self.accuracy)
        self.matching_score = np.mean(self.matching_score)
        print('task: ', self.params['task_type'])
        if self.params['task_type'] == 'repeatability':
            rep = torch.as_tensor(self.repeatability).cpu().numpy()
            plot_repeatability(rep, self.params['repeatability_params']['save_path'])
            rep = np.mean(rep)

            error = torch.as_tensor(self.rep_mean_err).cpu().numpy()
            error = error[~np.isnan(error)]
            plot_repeatability(error, self.params['repeatability_params']['save_path'].replace('.png', '_error.png'))
            error = np.mean(error)
            print('repeatability', rep, ' rep_mean_err', error)

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:

        warp01_params = {}
        warp10_params = {}
        if 'warp01_params' in batch:
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[0]
            for k, v in batch['warp10_params'].items():
                warp10_params[k] = v[0]

        # pairs dataset
        score_map_0 = None
        score_map_1 = None
        desc_map_0 = None
        desc_map_1 = None

        # image pair dataset
        if batch['dataset'][0] == 'HPatches' or \
           batch['dataset'][0] == 'megaDepth' or \
           batch['dataset'][0] == 'image_pair':
            result0 = self.model(batch['image0'])
            result1 = self.model(batch['image1'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()

        # sequence dataset
        last_img = None

        if batch['dataset'][0] == 'Kitti' or batch['dataset'][0] == 'Euroc' or batch['dataset'][0] == 'TartanAir':
            if self.last_batch is None:
                self.last_batch = batch
            result0 = self.model(self.last_batch['image0'])
            result1 = self.model(batch['image0'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()
            last_img = self.last_batch['image0']
            self.last_batch = batch
        # task
        result = None
        if self.params['task_type'] == 'repeatability':
            result = repeatability(batch_idx, batch['image0'], score_map_0,
                                   batch['image1'], score_map_1,
                                   warp01_params, warp10_params, self.params)
            self.num_feat.append(result['num_feat'])
            self.repeatability.append(result['repeatability'])
            self.rep_mean_err.append(result['mean_error'])

        return result
