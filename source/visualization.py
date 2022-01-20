import ipdb
import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis(object):
    def __init__(self, log_dir):

        self.exp_path = log_dir
        if not (os.path.isdir(self.exp_path)):
            print(f"Making tensorboard log dir {self.exp_path}")
            os.mkdir(self.exp_path)
        self.writer = SummaryWriter(self.exp_path)

    def update_writer(self, update):
        ''' update the summary writer'''

        # update all scalars
        for scalar in update["scalar"]:
            self.writer.add_scalar(**scalar)

        self.writer.flush()

    def close_writer(self):
        ''' close the summary writer'''
        self.writer.close()
