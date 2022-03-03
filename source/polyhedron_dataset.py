
import ipdb
import numpy as np
import os
import pickle
import plotly.graph_objects as pgo
from torch.utils.data import Dataset
import trimesh

import source.polyhedron_utils as pu


class PolyhedronDataSet(Dataset):

    def __init__(self, pc_type, data_dir, transform=None, noise_scale=None):
        ''' initialize polyhedron dataset'''

        self.data_dir = data_dir
        self.transform = transform
        self.noise_scale = noise_scale

        self.data = []
        self.obj_paths = []
        self.vertices = []
        self.faces = []
        self.nfaces = []

        for dir in sorted(os.listdir(self.data_dir)):
            dir_path = os.path.join(data_dir, dir)
            if os.path.isdir(dir_path):

                # all info (i.e., pickle) files
                info_files = sorted([file for file in os.listdir(dir_path) if
                                     os.path.splitext(file)[1] == '.pickle'])

                # get polygon label from info files
                for file in info_files:

                    with open(os.path.join(dir_path, file), 'rb') as handle:
                        polyhedron_info = pickle.load(handle)

                    polyhedron_dir_path = os.path.join(
                        self.data_dir,
                        os.path.splitext(file)[0])

                    self.nfaces.append(polyhedron_info['n_faces'])

                    # compute vertices and faces
                    obj_file_name = polyhedron_info['obj_path'].split('/')[-1]
                    obj_path = os.path.join(dir_path, obj_file_name)
                    mesh = trimesh.load_mesh(file_obj=obj_path)
                    trimesh.repair.fix_winding(mesh)

                    self.obj_paths.append(obj_path)
                    self.vertices.append(np.array(mesh.vertices))
                    self.faces.append(np.array(mesh.faces))

                    if pc_type == 'drake_point_cloud':
                        drake_pc_path = os.path.join(
                            polyhedron_dir_path,
                            os.path.splitext(file)[0] + '_drake_pc.npy')
                        self.data.append(drake_pc_path)
                    elif pc_type == 'ideal_point_cloud':
                        ideal_pc_path = os.path.join(
                            polyhedron_dir_path,
                            os.path.splitext(file)[0] + '_pc.npy')
                        self.data.append(ideal_pc_path)
                    else:
                        raise RuntimeError("incorrect point cloud type")

        ct, self.class_dict, self.label_dict = 0, {}, {}
        for nface in sorted(self.nfaces):
            if not nface in self.class_dict:
                self.class_dict[nface] = ct
                self.label_dict[ct] = nface
                ct += 1

        self.labels = [self.class_dict[nface] for nface in self.nfaces]

    def __len__(self):
        ''' length of data set'''
        return len(self.labels)

    def __getitem__(self, idx):
        ''' return an item from the dataset: pointcloud, label, trimesh obj'''

        pointcloud = np.load(self.data[idx])
        label = self.labels[idx]
        vertices = self.vertices[idx]
        faces = self.faces[idx]
        T = None

        if self.transform:
            pointcloud, T, t = self.transform(pointcloud=pointcloud)
            vertices = np.dot(T, vertices.T).T + t

        if self.noise_scale:
            pointcloud = pu.gaussian_noise(pointcloud=pointcloud,
                                           scale=self.noise_scale)

        return {'pc': pointcloud, 'vrts': vertices, 'fcs': faces, 'lbl': label,
                'T': [T, t], 'pc_path': self.data[idx]}

    def collate_fn(self, batch):
        ''' collate function for this dataset '''
        return {
            'pc': np.stack([sample['pc'] for sample in batch], axis=0),
            'lbl': np.array([sample['lbl'] for sample in batch]),
            'vrts': [sample['vrts'] for sample in batch],
            'fcs': [sample['fcs'] for sample in batch],
            'T': [sample['T'] for sample in batch],
            'pc_path':  [sample['pc_path'] for sample in batch]
        }

    def get_nsides_from_labels(self, labels):
        ''' get number of sides from polygon labels'''

        if isinstance(labels, list):
            return [self.label_dict[int(label)] for label in labels]
        else:
            return self.label_dict[int(labels)]

    def plot_sample(self, idx, plot_opts={'plot_mesh': False}):
        ''' plot pointcloud sample. Also plot mesh if true'''

        sample = self.__getitem__(idx)

        data = [self.pointcloud_scatter(sample['pc'],
                                        color='rgba(0, 255, 0, 1)')]

        if plot_opts['plot_mesh']:
            data.append(self.plot_mesh(vertices=sample['vrts'],
                                       faces=sample['fcs'], color='gray'))

        return data

    def pointcloud_scatter(self, pointcloud, color, size=1):
        ''' plot a point cloud'''

        # pointclouds = pointclouds.numpy()

        data = pgo.Scatter3d(x=pointcloud[:, 0],
                             y=pointcloud[:, 1],
                             z=pointcloud[:, 2],
                             mode='markers',
                             marker_size=size,
                             marker_symbol='circle',
                             marker_color=color)
        return data

    def plot_mesh(self, vertices, faces, color):
        ''' plotting function '''

        # plot faces of trimesh objects
        data = pgo.Mesh3d(x=vertices[:, 0],
                          y=vertices[:, 1],
                          z=vertices[:, 2],
                          color=color,
                          opacity=0.8,
                          i=faces[:, 0],
                          j=faces[:, 1],
                          k=faces[:, 2])

        return data


# if __name__ == "__main__":

#     pc_type = 'ideal_point_cloud'
#     data_dir = '/home/nddoshi/Research/learning_sandbox/datasets/PolyhedronLowDimSmall/train'

#     dataset = PolyhedronDataSet(
#         pc_type=pc_type,
#         data_dir=data_dir,
#         transform=pu.train_transforms_3DRot(noise_scale=0.00)
#     )

#     sample_ind = np.random.randint(0, dataset.__len__()-1)
#     data = dataset.plot_sample(sample_ind)

#     fig = pgo.Figure(data)
#     fig.show()
