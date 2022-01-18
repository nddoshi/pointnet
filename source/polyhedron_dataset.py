
import ipdb
import numpy as np
import os
import pickle
import plotly.graph_objects as pgo
from torch.utils.data import Dataset

import source.polyhedron_utils


class PolyhedronDataSet(Dataset):

    def __init__(self, pc_type, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform

        self.data = []
        self.labels = []

        for dir in sorted(os.listdir(self.data_dir)):
            dir_path = os.path.join(data_dir, dir)
            if os.path.isdir(dir_path):

                # all info files
                info_files = sorted([file for file in os.listdir(dir_path) if
                                     os.path.splitext(file)[1] == '.pickle'])

                # get polygon label
                for file in info_files:
                    # print(os.path.splitext(file)[0])
                    # load file
                    with open(os.path.join(dir_path, file), 'rb') as handle:
                        polyhedron_info = pickle.load(handle)

                        polyhedron_dir_path = os.path.join(self.data_dir,
                                                           '_'.join(file.split('_')[:2]))

                        self.labels.append(polyhedron_info['n_faces'])

                        if pc_type == 'drake_point_cloud':
                            drake_pc_path = os.path.join(polyhedron_dir_path,
                                                         os.path.splitext(file)[0] + '_drake_pc.npy')
                            self.data.append(drake_pc_path)
                        elif pc_type == 'ideal_point_cloud':
                            ideal_pc_path = os.path.join(polyhedron_dir_path,
                                                         os.path.splitext(file)[0] + '_pc.npy')
                            self.data.append(ideal_pc_path)
                        else:
                            raise RuntimeError("incorrect point cloud type")

        ct, self.class_dict = 0, {}
        for label in self.labels:
            if not label in self.class_dict:
                self.class_dict[label] = ct
                ct += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        pointcloud = np.load(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            pointcloud = self.transform(pointcloud)

        return {'pointcloud': pointcloud,
                'class': self.class_dict[label]}

    def plot_sample(self, idx):
        ''' plot sample'''

        pointcloud = np.load(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            pointcloud = self.transform(pointcloud)

        # plot sampled points
        data = [pgo.Scatter3d(x=pointcloud[:, 0],
                              y=pointcloud[:, 1],
                              z=pointcloud[:, 2],
                              name=label,
                              mode='markers',
                              marker_size=4,
                              marker_symbol='circle',
                              marker_color='rgba(0, 0, 255, 1)')]

        return data


# if __name__ == "__main__":

#     pc_type = 'ideal_point_cloud'
#     data_dir = '/home/nddoshi/Research/learning_sandbox/datasets/Polygon'

#     dataset = PolyhedronDataSet(
#         pc_type=pc_type, data_dir=data_dir,
#         transform=polyhedron_utils.train_transforms(scale=0.02))

#     sample_ind = np.random.randint(0, dataset.__len__()-1)
#     sample = dataset.__getitem__(sample_ind)
#     plot_data = dataset.plot_sample(sample_ind)

#     figure = pgo.Figure(plot_data)
#     figure.show()
