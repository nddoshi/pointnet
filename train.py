import argparse
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
from source.args import parse_train_args
from torchvision import transforms
from torch.utils.data import DataLoader

from source.visualization import TensorBoardVis

random.seed = 42


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
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
    print("In training")
    path = Path(args.root_dir)
    print(path)
    folders = [dir for dir in sorted(
        os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

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
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)

    train_ds = dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = dataset.PointCloudData(
        path, valid=True, folder='test', transform=train_transforms)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))

    train_loader = DataLoader(
        dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size*2)

    try:
        os.mkdir(args.save_model_path)
    except OSError as error:
        print(error)

    # tensorboard visualization
    tensorboard_vis = TensorBoardVis({'log_dir': '../tensorboard-logs'})

    print('Start training')
    t0 = time.time()
    Nprint = 5
    step = 0
    for epoch in range(args.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(
                device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
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
        pointnet.eval()
        correct = total = 0

        # validation
        if valid_loader:
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(
                        device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            # build tensorboard update
            scalar_update_list = build_tensorboard_scalars(
                tags=['Validation Accuracy'], scalars=[val_acc], steps=[step])

            tensorboard_vis.update_writer({'scalar': scalar_update_list})
            print('Valid accuracy: %d %%' % val_acc)
        # save the model

        checkpoint = Path(args.save_model_path)/'save_'+str(epoch)+'.pth'
        torch.save(pointnet.state_dict(), checkpoint)
        print('Model saved to ', checkpoint)


if __name__ == '__main__':
    print("In the main function")
    args = parse_train_args()
    print("Got the args, calling training")
    train(args)
