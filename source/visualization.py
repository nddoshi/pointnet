import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis(object):
    def __init__(self, params):
        self.writer = SummaryWriter(**params)

    def update_writer(self, update):
        ''' update the summary writer'''

        # update all scalars
        for scalar in update["scalar"]:
            self.writer.add_scalar(**scalar)

        self.writer.flush()

    def close_writer(self):
        ''' close the summary writer'''
        self.writer.close()
