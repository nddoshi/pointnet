import ipdb
import math
import random
import os
import time
import torch
import numpy as np

from path import Path
from source import model
from source import dataset
from source import utils
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader

from source.args import parse_test_args

random.seed = 42


def test(args):

    path = Path(args.root_dir)

    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    pointnet = model.PointNet()
    pointnet.to(device)

    # load test model
    if device.type == 'cpu':
        pointnet.load_state_dict(torch.load(
            args.test_model_path, map_location=torch.device('cpu')))
    elif device.type == 'cuda':
        pointnet.load_state_dict(torch.load(args.test_model_path))
    else:
        raise RuntimeError("Device type not supported")

    pointnet.eval()

    valid_ds = dataset.PointCloudData(path, valid=True, folder='test',
                                      transform=train_transforms)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size*2)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
            inputs, labels = data['pointcloud'].float(), data['category']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, __, __ = pointnet(inputs.transpose(1, 2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


if __name__ == "__main__":
    args = parse_test_args()
    test(args)
