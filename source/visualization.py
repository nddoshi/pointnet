import datetime
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter


def plot_confusion_matrix(dataset, preds, true_vals):

    # classes
    pred_sides = [dataset.label_dict[pred] for pred in preds]
    true_sides = [dataset.label_dict[tv] for tv in true_vals]

    # unique classes
    unique_sides = sorted(list(set(pred_sides + true_sides)))

    # class labels
    classes = [f"Sides: {ns}" for ns in unique_sides]

    # confusion matrix
    cm = confusion_matrix(true_vals, preds)
    df_cm = pd.DataFrame(cm/np.sum(cm), index=[i for i in classes],
                         columns=[i for i in classes])

    # plot
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True, cmap='Greens').get_figure()


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
        if 'scalar' in update:
            for scalar in update['scalar']:
                self.writer.add_scalar(**scalar)

        if 'figure' in update:
            for figure in update['figure']:
                self.writer.add_figure(**figure)

        if 'mesh' in update:
            for mesh in update['mesh']:
                self.writer.add_mesh(**mesh)

        self.writer.flush()

    def close_writer(self):
        ''' close the summary writer'''
        self.writer.close()
