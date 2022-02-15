import copy
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.tensorboard import SummaryWriter


def plot_confusion_matrix(dataset, preds, true_vals):
    ''' plot confusion matrix with true on the rows and preds on the columns '''

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


def build_tensorboard_scalars(tags, scalars, steps):
    ''' build scalars for tensorboard'''

    assert len(tags) == len(scalars) == len(steps)

    scalar_updates = []
    for tag, scalar, step in zip(tags, scalars, steps):
        scalar_updates.append({
            'tag': tag, 'scalar_value': scalar, 'global_step': step
        })

    return scalar_updates


def build_tensorboard_meshes(tag, xyz, face, vertices, crit_pt_ind, color,
                             global_step, options={'all_pts': False, 'mesh': True}):
    ''' build tensorboard mesh updates'''

    mesh_updates = []

    # make critical pt mask
    crit_pts_inds_unique = np.unique(crit_pt_ind)
    crit_pts_mask = np.full(xyz.shape[0], False, dtype=bool)
    crit_pts_mask[crit_pts_inds_unique] = True

    # colors for plotting only critical points
    colors = np.array([[255, 255, 255]] * xyz.shape[0])
    colors[crit_pts_inds_unique, :] = np.array(
        [color] * crit_pts_inds_unique.shape[0])

    # always plot critical points
    point_update = {'tag': tag,
                    'vertices': xyz,
                    'colors': colors,
                    'global_step': global_step}

    if options['all_pts']:  # add all points

        # update colors for plotting all points
        colors[~crit_pts_mask] = np.array(
            [[200, 200, 200]] * (xyz.shape[0] - crit_pts_inds_unique.shape[0]))

        point_update = {'tag':  tag,
                        'vertices': xyz,
                        'colors': colors,
                        'global_step': global_step}

    if options['mesh']:  # add mesh

        point_update['vertices'] = np.vstack([vertices, xyz])
        point_update['colors'] = np.vstack(
            [np.array([[0, 0, 255]] * vertices.shape[0]), colors])

        mesh_update = {'tag': tag,
                       'vertices': vertices,
                       'faces': face,
                       'colors': 200 * np.ones(shape=vertices.shape),
                       'global_step': global_step + 1}

        mesh_updates.append(mesh_update)

    mesh_updates.append(point_update)
    return mesh_updates


class TensorBoardVis(object):
    def __init__(self, log_dir, device):

        self.exp_path = log_dir
        self.device = device
        if not (os.path.isdir(self.exp_path)):
            print(f"Making tensorboard log dir {self.exp_path}")
            os.mkdir(self.exp_path)
        self.writer = SummaryWriter(self.exp_path)
        self.camera_config = {
            'cls': 'PerspectiveCamera',
            'fov': 75,
            'aspect': 0.9,
        }

    def update_writer(self, update):
        ''' update the summary writer'''

        # update all scalars
        if 'scalar' in update:
            for scalar in update['scalar']:
                self.writer.add_scalar(**scalar)
        # update all figures
        if 'figure' in update:
            for figure in update['figure']:
                self.writer.add_figure(**figure)
        # update all meshes
        if 'mesh' in update:
            for mesh in update['mesh']:
                self.writer.add_mesh(
                    **self.mesh_update_to_tensor(mesh_update=mesh),
                    config_dict=self.camera_config)

        self.writer.flush()

    def mesh_update_to_tensor(self, mesh_update):

        mesh_update_tensor = copy.deepcopy(mesh_update)

        mesh_update_tensor['vertices'] = torch.as_tensor(
            mesh_update['vertices'], dtype=torch.float,
            device=self.device).unsqueeze(0)
        mesh_update_tensor['colors'] = torch.as_tensor(
            mesh_update['colors'], dtype=torch.int,
            device=self.device).unsqueeze(0)

        try:
            mesh_update_tensor['faces'] = torch.as_tensor(
                mesh_update['faces'], dtype=torch.int,
                device=self.device).unsqueeze(0)
        except KeyError:
            pass

        return mesh_update_tensor

    def close_writer(self):
        ''' close the summary writer'''
        self.writer.close()
