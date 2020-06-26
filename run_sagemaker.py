import argparse
import os

import sagemaker
from sagemaker.tensorflow import TensorFlow


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# for this file
parser = argparse.ArgumentParser(description='Training MAML')
parser.add_argument('--instance_type', type=str, default='ml.p3.2xlarge')
parser.add_argument('--data_path', type=str, help='s3', default='s3://scsk-data/lionel/maml/data')
parser.add_argument('--output_path', type=str, help='path to store to s3',
                    default='s3://scsk-data/lionel/maml/output')
parser.add_argument('--checkpoint_path', type=str,
                    help='s3 path to store checkpoint when training with spot instance',
                    default='s3://scsk-data/lionel/maml/checkpoints')
parser.add_argument('--job_name', type=str, help='job name when training on sagemaker', default='maml')

# for train_sagemaker.py file
## Dataset/method options
parser.add_argument('--datasource', type=str, help='sinusoid or omniglot or miniimagenet', default='sinusoid')
parser.add_argument('--num_classes', type=int,
                    help='number of classes used in classification (e.g. 5-way classification).', default=5)
# oracle means task id is input (only suitable for sinusoid)
parser.add_argument('--baseline', type=str, help='oracle, or None', default=None)

## Training options
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

## Model options
parser.add_argument('--norm', type=str, help='batch_norm, layer_norm, or None', default='batch_norm')
parser.add_argument('--num_filters', type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.', default=64)
parser.add_argument('--conv', type=str2bool,
                    help='whether or not to use a convolutional network, only applicable in some cases', default='true')
parser.add_argument('--max_pool', type=str2bool,
                    help='Whether or not to use max pooling rather than strided convolutions', default='False')
parser.add_argument('--stop_grad', type=str2bool,
                    help='if True, do not use second derivatives in meta-optimization (for speed)', default='False')

## Logging, saving, and testing options
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

args = parser.parse_args()

os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
sess = sagemaker.Session()
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
account = '533155507761'
region = 'us-west-2'

# Define space of disk to storage input data
storage_space = 200  # Gb

# Maximum seconds for this training job’s life (days * hours * seconds)
train_max_run = 1 * 24 * 3600
train_max_wait = 1 * 24 * 3600

hyperparameters = {
    'datasource': str(args.datasource),
    'num_classes': str(args.num_classes),
    'baseline': str(args.baseline),
    'pretrain_iterations': str(args.pretrain_iterations),
    'metatrain_iterations': str(args.metatrain_iterations),
    'meta_batch_size': str(args.meta_batch_size),
    'meta_lr': str(args.meta_lr),
    'update_batch_size': str(args.update_batch_size),
    'update_lr': str(args.update_lr),
    'num_updates': str(args.num_updates),
    'norm': str(args.norm),
    'num_filters': str(args.num_filters),
    'conv': str(args.conv),
    'max_pool': str(args.max_pool),
    'stop_grad': str(args.stop_grad),
    'log': str(args.log),
    'logdir': str(args.logdir),
    'resume': str(args.resume),
    'train': str(args.train),
    'test_iter': str(args.test_iter),
    'test_set': str(args.test_set),
    'train_update_batch_size': str(args.train_update_batch_size),
    'train_update_lr': str(args.train_update_lr)
}

estimator = TensorFlow(
    entry_point='train_sagemaker.py',
    source_dir='.',
    code_location='s3://scsk-data/lionel/maml/code/',
    base_job_name=args.job_name,
    role=role,
    input_mode='File',
    train_instance_count=1,
    train_volume_size=storage_space,
    train_instance_type=args.instance_type,
    output_path=args.output_path,
    checkpoint_s3_uri=args.checkpoint_path,
    train_use_spot_instances=True,
    train_max_run=train_max_run,
    train_max_wait=train_max_wait,
    framework_version='1.14.0',
    py_version="py3",
    sagemaker_session=sess,
    hyperparameters=hyperparameters)

estimator.fit({'train': args.data_path})
