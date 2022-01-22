import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from torch.utils.data import DataLoader

from source.args import parse_test_args
from source import model
from source import polyhedron_dataset
from source import polyhedron_utils
from source import save_utils
from source import train_utils
from source import visualization

random.seed = 42


if __name__ == "__main__":
    # arguments
    args = parse_test_args()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    # load
    checkpoint, args = save_utils.load_experiment(args=args)

    # load training dataset to get num classes (TODO: fix this HACK)
    train_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args['point_cloud_type'],
        data_dir=os.path.join(args['dataset_dir'], 'train'),
        transform=polyhedron_utils.default_transforms())

    # testing dataset
    test_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args['point_cloud_type'],
        data_dir=os.path.join(args['dataset_dir'], 'test'),
        transform=polyhedron_utils.default_transforms())
    test_loader = DataLoader(dataset=test_ds, batch_size=len(test_ds),
                             shuffle=True)

    # print dateset info
    # ipdb.set_trace()
    num_classes = len(np.unique(np.array(train_ds.labels)))
    print(f'Valid dataset size: {len(test_ds)}')
    print(f'Number of classes: {num_classes}')

    # model
    pointnet = model.PointNet(classes=num_classes)
    # try:
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    # except RuntimeError:

    pointnet.to(device)
    pointnet.eval()

    step = 0
    print("Testing...")
    loss, acc, inputs, predictions, labels = train_utils.test_loop(
        dataloader=test_loader,
        train_dataset=train_ds,
        model=pointnet,
        lossfn=train_utils.pointnetloss,
        device=device)

    cm_fig = visualization.plot_confusion_matrix(
        dataset=train_ds, preds=predictions, true_vals=labels)

    print("Done!")
    plt.show()
