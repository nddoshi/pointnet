import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader

from source import model
from source import polyhedron_dataset
from source import polyhedron_utils
from source import train_utils
from source.args import parse_test_args

random.seed = 42


if __name__ == "__main__":
    # arguments
    args = parse_test_args()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    # testing dataset
    test_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args.point_cloud_type,
        data_dir=os.path.join(args.dataset_dir, 'test'),
        transform=polyhedron_utils.default_transforms())
    valid_loader = DataLoader(dataset=test_ds, batch_size=len(test_ds),
                              shuffle=True)

    # print dateset info
    num_classes = len(np.unique(np.array(test_ds.labels)))
    print(f'Valid dataset size: {len(test_ds)}')
    print(f'Number of classes: {num_classes}')

    # model
    pointnet = model.PointNet()
    pointnet.to(device)
    pointnet.eval()

    if device.type == 'cpu':
        pointnet.load_state_dict(torch.load(
            args.test_model_path, map_location=torch.device('cpu')))
    elif device.type == 'cuda':
        pointnet.load_state_dict(torch.load(args.test_model_path))
    else:
        raise RuntimeError("Device type not supported")

    step = 0
    print("Testing...")
    inputs, predictions, labels = train_utils.test_loop(dataloader=valid_loader,
                                                        model=pointnet,
                                                        lossfn=train_utils.pointnetloss,
                                                        device=device)

    print("Done!")
