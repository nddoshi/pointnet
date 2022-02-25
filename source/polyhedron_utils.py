import ipdb
import numpy as np
import scipy as scp


def rotation_z(pointcloud):
    ''' random rotation about the z-axis'''

    assert len(pointcloud.shape) == 2

    theta = np.random.rand() * 2. * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta),    0],
                  [np.sin(theta),  np.cos(theta),    0],
                  [0,                             0,      1]])

    rot_pointcloud = R.dot(pointcloud.T).T
    return rot_pointcloud, R, np.zeros(pointcloud.shape[1])


def rotation_S03(pointcloud):
    ''' random rotation form S03 '''

    assert len(pointcloud.shape) == 2
    R = scp.spatial.transform.Rotation.random().as_matrix()
    return R.dot(pointcloud.T).T, R, np.zeros(pointcloud.shape[1])


def normalize(pointcloud):
    assert len(pointcloud.shape) == 2

    pc_mean = np.mean(pointcloud, axis=0)
    norm_pointcloud = pointcloud - pc_mean

    max_pt_norm = np.max(np.linalg.norm(norm_pointcloud, axis=1))
    norm_pointcloud /= max_pt_norm
    Tnorm = (1./max_pt_norm) * np.identity(len(pc_mean))

    return norm_pointcloud, Tnorm, -np.dot(Tnorm, pc_mean)


def gaussian_noise(pointcloud, scale=0.02, noise_prob=0.95):
    assert len(pointcloud.shape) == 2

    if np.random.rand() < noise_prob:
        noise = np.random.normal(0, scale, (pointcloud.shape))
        pointcloud += noise

    return pointcloud


def train_transforms_3DRot(pointcloud):
    ''' normalized and rotate by random SO3 rotation'''

    normalized_pc, T1, t1 = normalize(pointcloud=pointcloud)
    rotated_normalized_pc, T2, t2 = rotation_S03(pointcloud=normalized_pc)

    return rotated_normalized_pc, np.dot(T2, T1), np.dot(T2, t1) + t2


def default_transforms(pointcloud):
    ''' normalize '''

    return normalize(pointcloud=pointcloud)


# def train_transforms(pointcloud):

    # return transforms.Compose([
    #     Normalize(),
    #     RandRotation_z(),
    #     RandomNoise(scale=noise_scale),
    # ])
    # class Normalize(object):
    #     def __call__(self, pointcloud):
    #         assert len(pointcloud.shape) == 2

    #         norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
    #         norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

    #         return norm_pointcloud

    # class RandRotation_z(object):
    #     def __call__(self, pointcloud):
    #         assert len(pointcloud.shape) == 2

    #         theta = random.random() * 2. * math.pi
    #         rot_matrix = np.array([[math.cos(theta), -math.sin(theta),    0],
    #                                [math.sin(theta),  math.cos(theta),    0],
    #                                [0,                             0,      1]])

    #         rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    #         return rot_pointcloud

    # class RandRotation_S03(object):

    #     def __call__(self, pointcloud):

    #         assert len(pointcloud.shape) == 2

    #         R = scp.spatial.transform.Rotation.random().as_matrix()
    #         return R.dot(pointcloud.T).T

    # class RandomNoise(object):
    #     def __init__(self, scale=0.02):
    #         self.scale = scale

    #     def __call__(self, pointcloud):
    #         assert len(pointcloud.shape) == 2

    #         noise = np.random.normal(0, self.scale, (pointcloud.shape))

    #         noisy_pointcloud = pointcloud + noise
    #         return noisy_pointcloud
