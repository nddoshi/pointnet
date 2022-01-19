import argparse
import ipdb
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix
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
    criterion = torch.nn.NLLLoss()  # negative log likelihood loss
    batch_size = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(batch_size, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64-torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (
        torch.norm(diff3x3)+torch.norm(diff64x64)) / float(batch_size)


def build_tensorboard_scalars(tags, scalars, steps):
    ''' build scalars for tensorboard'''

    assert len(tags) == len(scalars) and len(scalars) == len(steps)

    scalar_updates = []
    for tag, scalar, step in zip(tags, scalars, steps):
        scalar_updates.append({
            'tag': tag, 'scalar_value': scalar, 'global_step': step
        })

    return scalar_updates


def train_loop(dataloader, model, lossfn, optimizer, device,
               tensorboard_vis, step):

    current = 0
    for batch, (X, y) in enumerate(dataloader):

        # get data
        inputs, labels = X.to(device).float(), y.to(device)

        # current prediction and loss
        outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
        loss = lossfn(outputs, labels, m3x3, m64x64)
        correct = (outputs.argmax(1) ==
                   labels).type(torch.float).sum().item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        # if batch % 10 == 0:
        loss = loss.item()
        current += len(X)

        # build tensorboard update
        scalar_update_list = build_tensorboard_scalars(
            tags=['Loss/train'], scalars=[loss], steps=[step + current])
        tensorboard_vis.update_writer({'scalar': scalar_update_list})

        print(
            f"loss: {loss:>7f}, accuracy: {100 * correct/len(labels)}%, [{current:>5d}/{len(dataloader.dataset):>5d}]")

    return step + current


def test_loop(dataloader, model, lossfn, device, tensorboard_vis, step):

    # all_labels, all_preds = [], []

    test_loss, correct = 0, 0
    size, num_batches = 0, 0
    # current = 0
    with torch.no_grad():
        for X, y in dataloader:
            # get data
            inputs, labels = X.to(device).float(), y.to(device)

            outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))
            test_loss += lossfn(outputs, labels, m3x3, m64x64).item()
            correct += (outputs.argmax(1) ==
                        labels).type(torch.float).sum().item()
            # all_labels.extend(labels.tolist())
            # all_preds.extend(outputs.argmax(1).tolist())
            # current += len(X)
            # print(
            # f"loss: {test_loss:>7f}, accurary: {100 * correct/len(labels)}%, [{current:>5d}/{len(dataloader.dataset):>5d}]")
            size += len(X)
            num_batches += 1

    test_loss /= num_batches
    correct /= size

    # build tensorboard update
    scalar_update_list = build_tensorboard_scalars(
        tags=['Loss/test', 'Validation Accuracy'], scalars=[test_loss, correct],
        steps=[step, step])
    tensorboard_vis.update_writer({'scalar': scalar_update_list})

    print(
        f"Test Error:  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    # arguments
    args = parse_train_args()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    # training dataset
    train_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args.point_cloud_type,
        data_dir=os.path.join(args.dataset_dir, 'train'),
        transform=polyhedron_utils.train_transforms(noise_scale=0.02))
    train_loader = DataLoader(
        dataset=train_ds, batch_size=args.batch_size, shuffle=True)

    # testing dataset
    valid_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args.point_cloud_type,
        data_dir=os.path.join(args.dataset_dir, 'train'),
        transform=polyhedron_utils.default_transforms())
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size,
                              shuffle=True)

    # print dateset info
    num_classes = len(np.unique(np.array(train_ds.labels)))
    print(f'Train dataset size: {len(train_ds)}')
    print(f'Valid dataset size: {len(valid_ds)}')
    print(f'Number of classes: {num_classes}')

    # model
    pointnet = model.PointNet(classes=num_classes)
    pointnet.to(device)

    # optimizer
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)

    # tensorboard visualization
    tensorboard_vis = TensorBoardVis(log_dir=args.tb_log_dir)

    # saving checkpoints
    # checkpoint_dir = os.path.join(
    #     args.save_model_path, tensorboard_vis.exp_name)
    # if not(os.path.isdir(checkpoint_dir)):
    #     os.mkdir(checkpoint_dir)

    step = 0
    test_loop(dataloader=valid_loader, model=pointnet, lossfn=pointnetloss,
              device=device, tensorboard_vis=tensorboard_vis, step=step)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        step = train_loop(dataloader=train_loader, model=pointnet, lossfn=pointnetloss,
                          optimizer=optimizer, device=device,
                          tensorboard_vis=tensorboard_vis, step=step)
        # print("----------------Test---------------")
        test_loop(dataloader=valid_loader, model=pointnet, lossfn=pointnetloss,
                  device=device, tensorboard_vis=tensorboard_vis, step=step)

        # save the model
        # checkpoint = os.path.join(
        #     checkpoint_dir, 'polyhedron_classification_save_'+str(epoch)+'.pth')
        # torch.save(pointnet.state_dict(), checkpoint)
        # print('Model saved to ', checkpoint)

    # tensorboard_vis.close_writer()
    print("Done!")
