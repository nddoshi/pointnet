import argparse
import ipdb
import numpy as np
import random
import os
import time
import torch
from torch.utils.data import DataLoader

from source import model
from source import polyhedron_dataset
from source import polyhedron_utils
from source.args import parse_train_args
from source.visualization import TensorBoardVis

random.seed = 42


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    ''' loss function '''

    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64-torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)


def build_tensorboard_scalars(tags, scalars, steps):
    ''' build scalars for tensorboard'''

    assert len(tags) == len(scalars) and len(scalars) == len(steps)

    scalar_updates = []
    for tag, scalar, step in zip(tags, scalars, steps):
        scalar_updates.append({
            'tag': tag, 'scalar_value': scalar, 'global_step': step
        })

    return scalar_updates


def train(args):

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    # dataset
    train_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args.point_cloud_type, data_dir=args.dataset_dir,
        transform=polyhedron_utils.train_transforms())
    train_loader = DataLoader(
        dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    num_classes = len(np.unique(np.array(train_ds.labels)))

    print(f'Train dataset size: {len(train_ds)}')
    print(f'Number of classes: {num_classes}')

    # model
    pointnet = model.PointNet(classes=num_classes)
    pointnet.to(device)

    # optimizer
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)

    # saving checkpoints
    if not(os.path.isdir(args.save_model_path)):
        os.mkdir(args.save_model_path)

    # tensorboard visualization
    tensorboard_vis = TensorBoardVis(log_dir=args.tb_log_dir)

    print('Start training')
    t0 = time.time()
    Nprint = 5
    step = 0
    for epoch in range(args.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get data
            inputs, labels = data['pointcloud'].to(
                device).float(), data['class'].to(device)

            # not sure
            optimizer.zero_grad()

            # current prediction
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            # loss
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()

            # optimize
            optimizer.step()

            # build tensorboard update
            scalar_update_list = build_tensorboard_scalars(
                tags=['Loss/train'], scalars=[loss.item()], steps=[step])
            tensorboard_vis.update_writer({'scalar': scalar_update_list})

            # print statistics
            running_loss += loss.item()
            if i % Nprint == Nprint-1:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f, Total Time: %.3f [min]' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / Nprint, (time.time() - t0)/60.))
                running_loss = 0.0

            step += 1

        # save the model
        checkpoint = os.path.join(
            args.save_model_path, 'polyhedron_classification_save_'+str(epoch)+'.pth')
        torch.save(pointnet.state_dict(), checkpoint)
        print('Model saved to ', checkpoint)

    tensorboard_vis.close_writer()


if __name__ == '__main__':
    args = parse_train_args()
    train(args)
