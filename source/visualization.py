import ipdb
import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis(object):
    def __init__(self, log_dir):

        exp_name = self.generate_experiment_name(log_dir)
        self.exp_path = os.path.join(log_dir, exp_name)

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

    def generate_experiment_name(self, log_dir):
        ''' generate experiment name'''

        prev_exps = os.listdir(log_dir)
        num_prev_exp = 1
        for prev_exp in prev_exps:
            prev_exp_path = os.path.join(log_dir, prev_exp)
            if os.path.isdir(prev_exp_path) and "exp" in prev_exp:
                num_prev_exp += 1

        date_string = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        return f"exp_{num_prev_exp:03d}_{date_string}"

    def close_writer(self):
        ''' close the summary writer'''
        self.writer.close()
