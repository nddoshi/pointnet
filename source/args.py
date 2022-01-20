import sys
import argparse


def parse_train_args():
    parser = argparse.ArgumentParser(description='')

    # experiment setup
    parser.add_argument('--exp_name', default='dummy-exp', type=str,
                        help='experiment name')
    parser.add_argument('--dataset_dir', default='../datasets/Polygon/', type=str,
                        help='dataset directory')
    parser.add_argument('--resume_epoch', default=0, type=int,
                        help='iteration to resume training on')
    parser.add_argument('--load_dir', default="", type=str,
                        help='directory of experiment to load')

    # hyper parameters
    parser.add_argument('--point_cloud_type', default='ideal_point_cloud', type=str,
                        help='ideal or drake point cloud')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of training epochs')
    parser.add_argument('--noise_scale', default=0.02, type=float,
                        help='scale for random noise to add')

    # settings related to saving stuff
    parser.add_argument('--save_flag', default=False, type=bool,
                        help='save experiment if true')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='freq of saving checkpoints (in epochs)')
    parser.add_argument('--tb_log_dir', default='../experiments/tensorboard-logs/', type=str,
                        help='tensorboard log directory')
    parser.add_argument('--save_dir', default='../experiments/', type=str,
                        help='checkpoints dir')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    parser.add_argument('--dataset_dir', default='../datasets/Polygon/', type=str,
                        help='dataset directory')
    parser.add_argument('--test_model_path', default='./checkpoints/exp_006_01-18-2022_18:57:00/polyhedron_classification_save_14.pth', type=str,
                        help='path of trained model')
    parser.add_argument('--point_cloud_type', default='ideal_point_cloud', type=str,
                        help='ideal or drake point cloud')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    return args
