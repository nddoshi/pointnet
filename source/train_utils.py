import ipdb
import numpy as np
import plotly.graph_objects as pgo
import torch

from source import visualization


def append_to_save_data(save_data, data):
    ''' append data to save data'''
    for key in save_data.keys():
        if key in data:
            save_data[key].extend(data[key])
    return save_data


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


def random_index_from_mask(mask):
    ''' select one index from mask'''

    # construct index where mask is True
    true_indices = np.where(mask == True)[0]

    # sample randomly from true indices
    return true_indices[np.random.randint(low=0, high=len(true_indices))]


def train_loop(dataloader, model, lossfn, optimizer, device,
               tensorboard_vis=None, step=0, debug=False):
    ''' single epoch of training '''

    num_samples = dataloader.batch_size * len(dataloader)  # total samples
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [-1] * num_samples, [-1] * num_samples
    save_data = {'pc_path': [], 'vrts': [],
                 'fcs': [],  'lbl': [], 'z': [], 'T': []}

    # random sampling for plotting
    rand_batch = np.random.randint(0, len(dataloader))

    for batch, data in enumerate(dataloader):

        # convert data to torch data
        inputs = torch.tensor(
            data['pc'], dtype=torch.float, device=device)
        labels = torch.tensor(
            data['lbl'],  dtype=torch.long, device=device)

        # current prediction and loss
        outputs, features, crit_pt_inds, m3x3, m64x64 = model(
            inputs.transpose(1, 2))
        crit_pt_inds = crit_pt_inds.detach().cpu().numpy()
        predicted_labels = outputs.argmax(1)
        loss = lossfn(outputs, labels, m3x3, m64x64)
        correct = (predicted_labels == labels).detach().cpu().numpy()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # append
        all_preds[batch * dataloader.batch_size: (
            batch+1) * dataloader.batch_size] = predicted_labels.tolist()
        if device.type == 'cpu':
            all_labels[batch * dataloader.batch_size: (
                batch+1) * dataloader.batch_size] = labels.tolist()
        else:
            all_labels[batch * dataloader.batch_size: (
                batch+1) * dataloader.batch_size] = labels.cpu().tolist()

        save_data = append_to_save_data(save_data=save_data, data=data)
        save_data['z'].extend(features.detach().cpu().numpy())

        # randomly sample correct/incorrect examples for point cloud viz
        if (batch == rand_batch) and tensorboard_vis and debug:

            mesh_update_list = []
            if True in correct:
                correct_sample = random_index_from_mask(mask=correct)
                tag_prefix = f"Train, {dataloader.dataset.get_nsides_from_labels(labels[correct_sample])} Faces/"

                mesh_update_list.extend(visualization.build_tensorboard_meshes(
                    tag=tag_prefix + "correct",
                    xyz=data['pc'][correct_sample, :, :],
                    face=data['fcs'][correct_sample],
                    vertices=data['vrts'][correct_sample],
                    crit_pt_ind=crit_pt_inds[correct_sample, :, :],
                    color=[0., 255., 0.],
                    global_step=step))

            if False in correct:
                incorrect_sample = random_index_from_mask(mask=~correct)
                tag_prefix = f"Train, {dataloader.dataset.get_nsides_from_labels(labels[incorrect_sample])} Faces/"

                mesh_update_list.extend(visualization.build_tensorboard_meshes(
                    tag=tag_prefix + "incorrect",
                    xyz=data['pc'][incorrect_sample, :, :],
                    face=data['fcs'][incorrect_sample],
                    vertices=data['vrts'][incorrect_sample],
                    crit_pt_ind=crit_pt_inds[incorrect_sample, :, :],
                    color=[255, 0., 0.],
                    global_step=step))

        # print batch statistics
        loss = loss.item()
        accuracy = np.sum(correct)/len(labels)
        print(
            f"loss: {loss:>7f}, accuracy: {100*accuracy}%, [{(batch+1)*len(inputs):>5d}/{len(dataloader.dataset):>5d}]")

        # cumulative statistics
        total_loss += loss
        total_correct += accuracy

    # build tensorboard scalar update once per epoch
    if tensorboard_vis:
        scalar_update_list = visualization.build_tensorboard_scalars(
            tags=['Loss/train per step', 'Accuracy/train per step'],
            scalars=[loss, accuracy],
            steps=[step + batch + 1, step + batch + 1])

        tensorboard_vis.update_writer({'scalar': scalar_update_list})

    # plot confusion matrix once per epoch
    # plot good/bad prediction meshes once per epoch
    if tensorboard_vis and debug:
        figure_update_list = [{
            'tag': 'Confusion Matrix/train per step',
            'figure': visualization.plot_confusion_matrix(
                dataset=dataloader.dataset, preds=all_preds,
                true_vals=all_labels),
            'global_step': step + batch + 1
        }]

        tensorboard_vis.update_writer(
            {'figure': figure_update_list,
             'mesh': mesh_update_list})

    return (total_loss/len(dataloader),
            total_correct/len(dataloader.dataset),
            save_data,
            step + batch + 1)


def test_loop(dataloader, train_dataset, model, lossfn, device,
              tensorboard_vis=None, step=0):
    ''' testing '''

    with torch.no_grad():
        for data in dataloader:

            # convert data to torch data
            inputs = torch.tensor(
                data['pc'], dtype=torch.float, device=device)
            labels = torch.tensor(
                data['lbl'],  dtype=torch.long, device=device)

            # predictions
            outputs, features, crit_pt_inds, m3x3, m64x64 = model(
                inputs.transpose(1, 2))
            crit_pt_inds = crit_pt_inds.detach().cpu().numpy()
            predictions = outputs.argmax(1)
            correct = (predictions == labels).detach().cpu().numpy()

            # loss and accuracy
            total_loss = lossfn(outputs, labels, m3x3, m64x64).item()
            total_correct = sum(correct)/len(dataloader.dataset)

    print(
        f"Test Error:  Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    if tensorboard_vis:
        # build tensorboard update for scalar
        scalar_update_list = visualization.build_tensorboard_scalars(
            tags=['Loss/test per step', 'Accuracy/test per step'],
            scalars=[total_loss, total_correct],
            steps=[step, step])

        # build tensorboard update for confusion matrix
        figure_update_list = [{
            'tag': 'Confusion Matrix/test per step',
            'figure': visualization.plot_confusion_matrix(
                dataset=train_dataset, preds=predictions.tolist(),
                true_vals=labels.tolist()),
            'global_step': step
        }]

        # build tensorboard mesh
        mesh_update_list = []
        if True in correct:

            correct_sample = random_index_from_mask(mask=correct)
            tag_prefix = f"Valid, {dataloader.dataset.get_nsides_from_labels(labels[correct_sample])} Faces/"

            mesh_update_list.extend(visualization.build_tensorboard_meshes(
                tag=tag_prefix + "correct",
                xyz=data['pc'][correct_sample, :, :],
                face=data['fcs'][correct_sample],
                vertices=data['vrts'][correct_sample],
                crit_pt_ind=crit_pt_inds[correct_sample, :, :],
                color=[0., 255., 0.],
                global_step=step))

        if False in correct:
            incorrect_sample = random_index_from_mask(mask=~correct)
            tag_prefix = f"Valid, {dataloader.dataset.get_nsides_from_labels(labels[incorrect_sample])} Faces/"

            mesh_update_list.extend(visualization.build_tensorboard_meshes(
                tag=tag_prefix + "incorrect",
                xyz=data['pc'][incorrect_sample, :, :],
                crit_pt_ind=crit_pt_inds[incorrect_sample, :, :],
                face=data['fcs'][incorrect_sample],
                vertices=data['vrts'][incorrect_sample],
                color=[255, 0., 0.],
                global_step=step))

        # update tensorboard
        tensorboard_vis.update_writer({'scalar': scalar_update_list,
                                       'figure': figure_update_list,
                                       'mesh': mesh_update_list})

    return (total_loss,
            total_correct,
            data,
            predictions.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            crit_pt_inds)
