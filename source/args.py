import sys
import argparse


def parse_train_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    parser.add_argument('--dataset_dir', default='../datasets/Polygon/', type=str,
                        help='dataset directory')
    parser.add_argument('--tb_log_dir', default='../tensorboard-logs/', type=str,
                        help='tensorboard log directory')
    parser.add_argument('--point_cloud_type', default='ideal_point_cloud', type=str,
                        help='ideal or drake point cloud')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of training epochs')
    parser.add_argument('--save_model_path', default='./checkpoints/', type=str,
                        help='checkpoints dir')

    args = parser.parse_args()

    assert args.dataset_dir is not None

    print(' '.join(sys.argv))

    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    parser.add_argument('--root_dir', default='../datasets/ModelNet10/', type=str,
                        help='dataset directory')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--test_model_path', default='./trained_models/latest.pth', type=str,
                        help='path of trained model')

    args = parser.parse_args()

    assert args.root_dir is not None

    print(' '.join(sys.argv))

    return args
