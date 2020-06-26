import argparse
import os
import shutil
import subprocess
import zipfile


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Training MAML')

# Dataset/method options
parser.add_argument('--datasource', type=str, help='sinusoid or omniglot or miniimagenet', default='sinusoid')
parser.add_argument('--num_classes', type=int,
                    help='number of classes used in classification (e.g. 5-way classification).', default=5)
# oracle means task id is input (only suitable for sinusoid)
parser.add_argument('--baseline', type=str, help='oracle, or None', default=None)

# Training options
parser.add_argument('--pretrain_iterations', type=int, help='number of pre-training iterations.', default=0)
parser.add_argument('--metatrain_iterations', type=int,
                    help='number of metatraining iterations. 15k for omniglot, 50k for sinusoid', default=15000)
parser.add_argument('--meta_batch_size', type=int, help='number of tasks sampled per meta-update', default=25)
parser.add_argument('--meta_lr', type=float, help='the base learning rate of the generator', default=0.001)
parser.add_argument('--update_batch_size', type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).', default=5)
parser.add_argument('--update_lr', type=float, help='step size alpha for inner gradient update., 0.1 for omniglot',
                    default=1e-3)
parser.add_argument('--num_updates', type=int, help='number of inner gradient updates during training.', default=1)

# Model options
parser.add_argument('--norm', type=str, help='batch_norm, layer_norm, or None', default='batch_norm')
parser.add_argument('--num_filters', type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.', default=64)
parser.add_argument('--conv', type=str2bool,
                    help='whether or not to use a convolutional network, only applicable in some cases', default='true')
parser.add_argument('--max_pool', type=str2bool,
                    help='Whether or not to use max pooling rather than strided convolutions', default='False')
parser.add_argument('--stop_grad', type=str2bool,
                    help='if True, do not use second derivatives in meta-optimization (for speed)', default='False')

# Logging, saving, and testing options
parser.add_argument('--log', type=str2bool,
                    help='if false, do not log summaries, for debugging code.', default='True')
parser.add_argument('--logdir', type=str, help='directory for summaries and checkpoints.',
                    default='/opt/ml/model/')
parser.add_argument('--resume', type=str2bool, help='resume training if there is a model available', default='True')
parser.add_argument('--train', type=str2bool, help='True to train, False to test.', default='True')
parser.add_argument('--test_iter', type=int, help='iteration to load model (-1 for latest model)', default=-1)
parser.add_argument('--test_set', type=str2bool,
                    help='Set to true to test on the the test set, False for the validation set.', default='False')
parser.add_argument('--train_update_batch_size', type=int,
                    help='number of examples used for gradient update during '
                         'training (use if you want to test with a different number).',
                    default=-1)
parser.add_argument('--train_update_lr', type=float,
                    help='value of inner gradient step step during training. '
                         '(use if you want to test with a different value)0.1 for omniglot',
                    default=-1)

# ignore error:
parser.add_argument('--model_dir', type=str, help='model_dir was automatic passed from run_sagemaker.py',
                    default='True')

args = parser.parse_args()

DATADIR = '/opt/ml/input/data/train/'


def init_env():
    subprocess.run("pip install -r requirements.txt", shell=True)


def preprocess():
    if args.datasource == 'omniglot':
        shutil.copy2(os.path.join(DATADIR, 'omniglot_resized.zip'), './data/')
        with zipfile.ZipFile('./data/omniglot_resized.zip', 'r') as zip_ref:
            zip_ref.extractall('./data/')
    elif args.datasource == 'miniimagenet':
        shutil.copy2(os.path.join(DATADIR, 'miniImagenet.zip'), './data/')
        with zipfile.ZipFile('./data/miniImagenet.zip', 'r') as zip_ref:
            zip_ref.extractall('./data/')
    elif args.datasource == 'sinusoid':
        print("Sinusoid is auto generated")
    else:
        raise ValueError('args.datasource must be in {omniglot, miniimagenet, sinusoid}')


def train():
    subprocess.run("python main.py "
                   "--datasource {} "
                   "--num_classes {} "
                   "--baseline {} "
                   "--pretrain_iterations {} "
                   "--metatrain_iterations {} "
                   "--meta_batch_size {} "
                   "--meta_lr {} "
                   "--update_batch_size {} "
                   "--update_lr {} "
                   "--num_updates {} "
                   "--norm {} "
                   "--num_filters {} "
                   "--conv {} "
                   "--max_pool {} "
                   "--stop_grad {} "
                   "--log {} "
                   "--logdir {} "
                   "--resume {} "
                   "--train {} "
                   "--test_iter {} "
                   "--test_set {} "
                   "--train_update_batch_size {} "
                   "--train_update_lr {} ".format(
        args.datasource,
        args.num_classes,
        args.baseline,
        args.pretrain_iterations,
        args.metatrain_iterations,
        args.meta_batch_size,
        args.meta_lr,
        args.update_batch_size,
        args.update_lr,
        args.num_updates,
        args.norm,
        args.num_filters,
        args.conv,
        args.max_pool,
        args.stop_grad,
        args.log,
        args.logdir,
        args.resume,
        args.train,
        args.test_iter,
        args.test_set,
        args.train_update_batch_size,
        args.train_update_lr), shell=True)


def evaluate():
    # TODO: run multiple test iter
    import glob
    model_names = glob.glob('**/*.index')
    iters = []

    for name in model_names:
        iters.append(int(name[5:-6]))
    iters.sort(key=lambda item: (-len(item), item))

    for i in iters:
        subprocess.run("python main.py "
                       "--datasource {} "
                       "--num_classes {} "
                       "--baseline {} "
                       "--pretrain_iterations {} "
                       "--metatrain_iterations {} "
                       "--meta_batch_size {} "
                       "--meta_lr {} "
                       "--update_batch_size {} "
                       "--update_lr {} "
                       "--num_updates {} "
                       "--norm {} "
                       "--num_filters {} "
                       "--conv {} "
                       "--max_pool {} "
                       "--stop_grad {} "
                       "--log {} "
                       "--logdir {} "
                       "--resume {} "
                       "--train {} "
                       "--test_iter {} "
                       "--test_set {} "
                       "--train_update_batch_size {} "
                       "--train_update_lr {} ".format(
            args.datasource,
            args.num_classes,
            args.baseline,
            args.pretrain_iterations,
            args.metatrain_iterations,
            args.meta_batch_size,
            args.meta_lr,
            args.update_batch_size,
            args.update_lr,
            args.num_updates,
            args.norm,
            args.num_filters,
            args.conv,
            args.max_pool,
            args.stop_grad,
            args.log,
            args.logdir,
            args.resume,
            'False',
            i,
            'True',
            args.train_update_batch_size,
            args.train_update_lr), shell=True)


def main():
    init_env()
    print("preprocessing")
    preprocess()
    print("training")
    train()


main()
