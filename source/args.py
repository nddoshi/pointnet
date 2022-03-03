import argparse
import sys


def parse_train_args():
    parser = argparse.ArgumentParser(description='')

    # experiment setup
    parser.add_argument('exp_name', default='dummy-exp', type=str,
                        help='experiment name')
    parser.add_argument('--dataset_dir', default='../datasets/Polygon/',
                        type=str, help='dataset directory')
    parser.add_argument('--resume_epoch', default=0, type=int,
                        help='iteration to resume training on')
    parser.add_argument('--load_dir', default="", type=str,
                        help='directory of experiment to load')

    # hyper parameters
    parser.add_argument('--random_seed', default=42, type=int,
                        help='seed for random number generator')
    parser.add_argument('--point_cloud_type', default='ideal_point_cloud',
                        type=str, help='ideal or drake point cloud')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of training epochs')
    parser.add_argument('--noise_scale', default=0.02, type=float,
                        help='scale for random noise to add')

    # settings related to saving stuff
    parser.add_argument('--save_flag', dest='save_flag', default=False,
                        action='store_true', help="save if true")
    parser.add_argument('--save_freq', default=5, type=int,
                        help='freq of saving checkpoints (in epochs)')
    parser.add_argument('--tb_log_dir', default='../experiments/tensorboard-logs/',
                        type=str, help='tensorboard log directory')
    parser.add_argument('--save_dir', default='../experiments/', type=str,
                        help='checkpoints dir')
    parser.add_argument('--tb_log_freq', default=5, type=int,
                        help='freq of updating tb log (in epochs)')
    parser.add_argument('--debug', dest='debug', default=False,
                        action='store_true', help="plot debug info if true")

    args = parser.parse_args()
    print(' '.join(sys.argv))
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings

    parser.add_argument("exp_name", type=str, help='name of exp to load')
    parser.add_argument('--dataset_dir', default='../datasets/PolyhedronLowDim/',
                        type=str, help='dataset directory')
    parser.add_argument(
        '--load_dir',
        default="/home/nddoshi/Dropbox (MIT)/pbal-affordance-assets/good_trials",
        type=str,
        help='directory of experiment to load')
    parser.add_argument('--resume_epoch', default=-1, type=int,
                        help='iteration to use for testing (default = -1 = last)')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    return args
