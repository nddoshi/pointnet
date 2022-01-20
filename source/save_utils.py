import datetime
import json
import ipdb
import os
import subprocess
import torch


def save_experiment(args):
    ''' save experiment '''

    # current date and time
    date_string = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    experiment_save_dir = os.path.join(
        args.save_dir, f"{args.exp_name}_{date_string}")
    tensorboard_save_dir = os.path.join(
        args.tb_log_dir, f"{args.exp_name}_{date_string}")

    # make directory for experiment
    if not os.path.isdir(experiment_save_dir):
        os.mkdir(experiment_save_dir)

    # make directory for tensorboard
    if not os.path.isdir(tensorboard_save_dir):
        os.mkdir(tensorboard_save_dir)

    # add commit has to args
    args_dict = vars(args)
    args_dict['commit'] = get_commit_hash()

    # save args
    with open(os.path.join(experiment_save_dir, 'args.txt'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    return experiment_save_dir, tensorboard_save_dir


def save_checkpoint(save_dir, epoch, model, optimizer, stats):

    stats["epoch"] = epoch
    stats['model_state_dict'] = model.state_dict()
    stats['opt_state_dict'] = optimizer.state_dict()

    checkpoint_fname = os.path.join(
        save_dir, f"model_{epoch}.pth")
    torch.save(stats, checkpoint_fname)
    print('Model saved to ', checkpoint_fname)


def get_commit_hash():
    """
    https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # Not copying .git folder into docker container
    except subprocess.CalledProcessError:
        commit = "0000000"
    # print(' > Git Hash: {}'.format(commit))
    return commit
