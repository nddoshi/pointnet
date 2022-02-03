import ipdb
import numpy as np
import torch

from source import visualization


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


def select_mesh_samples_to_plot(correct):
    ''' sample one correctly predict and one incorrectly predicted mesh to plot'''

    # construct index of correct/incorrectly predicted training data
    ind_correct = np.nonzero(correct.numpy())[0]
    ind_incorrect = np.nonzero((~correct).numpy())[0]

    # sample one of each
    correct_sample = np.random.choice(ind_correct)
    incorrect_sample = np.random.choice(ind_incorrect)

    return correct_sample, incorrect_sample


def build_tensorboard_scalars(tags, scalars, steps):
    ''' build scalars for tensorboard'''

    assert len(tags) == len(scalars) == len(steps)

    scalar_updates = []
    for tag, scalar, step in zip(tags, scalars, steps):
        scalar_updates.append({
            'tag': tag, 'scalar_value': scalar, 'global_step': step
        })

    return scalar_updates


def build_tensorboard_meshes(tags, xyzs, crit_pt_inds, colors, global_steps):
    ''' build tensorboard mesh updates'''

    assert len(tags) == len(xyzs) == len(
        crit_pt_inds) == len(colors) == len(global_steps)

    mesh_updates = []
    for tag, xyz, crit_pt_ind, color, global_step in zip(
            tags, xyzs, crit_pt_inds, colors, global_steps):

        pt_colors = xyz * 0 + torch.tensor(color)
        crit_pts = np.unique(crit_pt_ind)
        pt_colors[crit_pts, :] = torch.tensor([0.] * 3)
        mesh_updates.append(
            {'tag':  tag, 'vertices': xyz[None, :], 'colors': pt_colors[None, :],
             'global_step': global_step})

    return mesh_updates


def train_loop(dataloader, model, lossfn, optimizer, device,
               tensorboard_vis=None, step=0):
    ''' single epoch of training '''

    num_samples = dataloader.batch_size * len(dataloader)  # total samples
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [-1] * num_samples, [-1] * num_samples

    # random sampling for plotting
    rand_batch = np.random.randint(0, len(dataloader) - 1)

    for batch, (X, y) in enumerate(dataloader):

        # get data
        inputs, labels = X.to(device).float(), y.to(device)

        # current prediction and loss
        outputs, crit_pt_inds, m3x3, m64x64 = model(inputs.transpose(1, 2))
        predicted_labels = outputs.argmax(1)
        loss = lossfn(outputs, labels, m3x3, m64x64)
        correct = (predicted_labels == labels)

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

        # randomly sample correct/incorrect examples for point cloud viz
        if (batch == rand_batch) and tensorboard_vis:

            correct_sample, incorrect_sample = select_mesh_samples_to_plot(
                correct=correct)

            # build tensorboard mesh
            tag_prefix = f"Train, {dataloader.dataset.get_nsides_from_labels(all_labels[correct_sample])} Faces/"
            mesh_update_list = build_tensorboard_meshes(
                tags=[tag_prefix + "correct", tag_prefix + "incorrect"],
                xyzs=[inputs[correct_sample, :, :],
                      inputs[incorrect_sample, :, :]],
                crit_pt_inds=[crit_pt_inds[correct_sample, :, :],
                              crit_pt_inds[incorrect_sample, :, :]],
                colors=[[0., 255., 0, ], [255, 0., 0.]],
                global_steps=[step + batch + 1, step + batch + 1])

        # print batch statistics
        loss = loss.item()
        accuracy = correct.type(torch.float).sum().item()/len(labels)
        print(
            f"loss: {loss:>7f}, accuracy: {100*accuracy}%, [{(batch+1)*len(X):>5d}/{len(dataloader.dataset):>5d}]")

        # cumulative statistics
        total_loss += loss
        total_correct += accuracy

        # build tensorboard update
        if tensorboard_vis:
            scalar_update_list = build_tensorboard_scalars(
                tags=['Loss/train per step', 'Accuracy/train per step'],
                scalars=[loss, accuracy],
                steps=[step + batch + 1, step + batch + 1])

            tensorboard_vis.update_writer({'scalar': scalar_update_list})

    # plot confusion matrix once per epoch
    # plot good/bad prediction meshes once per epoch
    if tensorboard_vis:
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
            step + batch + 1)


def test_loop(dataloader, train_dataset, model, lossfn, device,
              tensorboard_vis=None, step=0):
    ''' testing '''

    with torch.no_grad():
        for X, y in dataloader:

            # get data
            inputs, labels = X.to(device).float(), y.to(device)

            # predictions
            outputs, crit_pt_inds, m3x3, m64x64 = model(inputs.transpose(1, 2))
            predictions = outputs.argmax(1)
            correct = (predictions == labels)

            # loss and accuracy
            total_loss = lossfn(outputs, labels, m3x3, m64x64).item()
            total_correct = correct.type(
                torch.float).sum().item()/len(dataloader.dataset)

    print(
        f"Test Error:  Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")

    if tensorboard_vis:
        # build tensorboard update for scalar
        scalar_update_list = build_tensorboard_scalars(
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
        correct_sample, incorrect_sample = select_mesh_samples_to_plot(
            correct=correct)

        tag_prefix = f"Valid, {dataloader.dataset.get_nsides_from_labels(labels[correct_sample])} Faces/"

        mesh_update_list = build_tensorboard_meshes(
            tags=[tag_prefix + "correct", tag_prefix + "incorrect"],
            xyzs=[inputs[correct_sample, :, :],
                  inputs[incorrect_sample, :, :]],
            crit_pt_inds=[crit_pt_inds[correct_sample, :, :],
                          crit_pt_inds[incorrect_sample, :, :]],
            colors=[[0., 255., 0, ], [255, 0., 0.]],
            global_steps=[step, step])

        # update tensorboard
        tensorboard_vis.update_writer({'scalar': scalar_update_list,
                                       'figure': figure_update_list,
                                       'mesh': mesh_update_list})

    return total_loss, total_correct, inputs, predictions, labels
