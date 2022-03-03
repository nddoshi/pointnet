import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.subplots as psp
import plotly.graph_objects as pgo
import random
import torch
from torch.utils.data import DataLoader
import trimesh

from source.args import parse_test_args
from source import analysis
from source import model
from source import polyhedron_dataset
from source import polyhedron_utils
from source import save_utils
from source import train_utils
from source import visualization
random.seed = 42


def compute_training_features(pc_path_list, transform_list, model, device,
                              datadir=None, max_size=1024):

    inputs = []
    for pc_path, T in zip(pc_path_list, transform_list):
        if datadir:
            pc_path = os.path.join(args.dataset_dir, *pc_path.split("/")[-3:])
        pc = np.load(pc_path)
        pc = np.dot(T[0], pc.T).T + T[1]
        inputs.append(pc)

    inds = np.random.randint(low=0, high=len(inputs), size=max_size)
    inputs = np.stack(inputs)[inds, :, :]
    with torch.no_grad():
        inputs = torch.tensor(inputs, dtype=torch.float, device=device)
        _, features, _, _, _ = model(inputs.transpose(1, 2))

    return features.detach().cpu().numpy(), inds


if __name__ == "__main__":
    # arguments
    args = parse_test_args()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")

    # load
    checkpoint_path, args_dict = save_utils.load_experiment(args=args)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # testing dataset
    test_ds = polyhedron_dataset.PolyhedronDataSet(
        pc_type=args_dict['point_cloud_type'],
        data_dir=os.path.join(args.dataset_dir, 'test'),
        transform=polyhedron_utils.default_transforms)

    test_loader = DataLoader(dataset=test_ds, batch_size=len(test_ds),
                             collate_fn=test_ds.collate_fn, shuffle=True)

    # print dateset info
    num_classes = len(np.unique(np.array(test_ds.labels)))
    print(f'Valid dataset size: {len(test_ds)}')
    print(f'Number of classes: {num_classes}')

    # model
    pointnet = model.PointNet(classes=num_classes)
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    pointnet.to(device)
    pointnet.eval()

    # compute training features and labels
    train_features, data_inds = compute_training_features(
        pc_path_list=checkpoint['data']['pc_path'],
        transform_list=checkpoint['data']['T'],
        model=pointnet,
        device=device,
        datadir=args.dataset_dir)
    train_labels = np.array(checkpoint['data']['lbl'])[data_inds]

    step = 0
    print("Testing...")
    loss, acc, data, predictions, features, crit_pt_inds = train_utils.test_loop(
        dataloader=test_loader,
        train_dataset=test_ds,
        model=pointnet,
        lossfn=train_utils.pointnetloss,
        device=device)
    correct = predictions == data['lbl']

    # unique classes
    unique_labels = np.unique(np.array(data['lbl']))
    num_labels = unique_labels.shape[0]

    # plot confusion matrix
    cm_fig = visualization.plot_confusion_matrix(
        dataset=test_ds, preds=predictions, true_vals=data['lbl'])
    cm_fig.get_axes()[0].set_ylabel('True Values')
    cm_fig.get_axes()[0].set_xlabel('Predictions')

    # compute knn
    feature_dist = np.zeros(
        shape=(train_features.shape[0], features.shape[0],))
    knn_ind, knn_label = [], []
    for i in range(features.shape[0]):

        feature_dist[:, i] = analysis.feature_distance_to_set(
            feature_set=train_features, query_feature=features[i, :])
        knn_ind.append(analysis.knn(
            feature_distance=feature_dist[:, i], k=4))
        knn_label.append(train_labels[knn_ind[-1]].tolist())

    true_faces = test_ds.get_nsides_from_labels(data['lbl'].tolist())
    knn_faces = [test_ds.get_nsides_from_labels(nn) for nn in knn_label]

    # plot knn
    knn_fig = visualization.plot_knn(true_faces=true_faces,
                                     knn_faces=knn_faces)

    # plot correct/incorrect examples
    fig = psp.make_subplots(
        rows=2, cols=num_labels,
        specs=[[{'type': 'surface'}
                for _ in range(num_labels)] for _ in range(2)])

    for i, label in enumerate(unique_labels):

        label_mask = data['lbl'] == label

        if True in correct * label_mask:

            # sample a correctly predicted shape
            correct_ind = train_utils.random_index_from_mask(
                mask=correct*label_mask)

            # plot critical points
            crit_pts_inds_unique = np.unique(crit_pt_inds[correct_ind, :, :])
            scatter = test_ds.pointcloud_scatter(
                pointcloud=data['pc'][correct_ind][crit_pts_inds_unique, :],
                color='green', size=2)
            fig.add_trace(scatter, row=1, col=i+1)

            # plot mesh
            mesh = test_ds.plot_mesh(vertices=data['vrts'][correct_ind],
                                     faces=data['fcs'][correct_ind],
                                     color="gray")
            fig.add_trace(mesh, row=1, col=i+1)

        if True in (~correct) * label_mask:

            # sample incorrectly predicted shape
            incorrect_ind = train_utils.random_index_from_mask(
                mask=(~correct)*label_mask)

            # plot critical points
            crit_pts_inds_unique = np.unique(crit_pt_inds[incorrect_ind, :, :])
            scatter = test_ds.pointcloud_scatter(
                pointcloud=data['pc'][incorrect_ind][crit_pts_inds_unique, :],
                color='red', size=2)
            fig.add_trace(scatter, row=2, col=i+1)

            # plot mesh
            mesh = test_ds.plot_mesh(vertices=data['vrts'][incorrect_ind],
                                     faces=data['fcs'][incorrect_ind],
                                     color="gray")
            fig.add_trace(mesh, row=2, col=i+1)

            print(
                f"Label {test_ds.get_nsides_from_labels(label)} incorrectly predicted as {test_ds.get_nsides_from_labels(predictions[incorrect_ind])}")

    print("Done!")

    # save figures
    save_path = os.path.join(args.load_dir, args.exp_name)
    knn_fig.write_html(os.path.join(save_path, 'knn'))
    cm_fig.savefig(os.path.join(save_path, 'cm.png'))

    fig.show()
    knn_fig.show()

    plt.show()
