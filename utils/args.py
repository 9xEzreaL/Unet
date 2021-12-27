#from optparse import OptionParser
from argparse import ArgumentParser

def merge_args(args, args_d):
    args.update(args_d)
    return args


def args_train():
    # Training Parameters
    parser = ArgumentParser()#OptionParser()
    # Name of the Project
    parser.add_argument('--prj', dest='prj', default='dess_segmentation',
                        type=str, help='name of the project')
    # How many epoch to run?
    parser.add_argument('-e', '--epochs', dest='epochs', default=200, type=int,
                        help='number of epochs')
    # Batch size per mini-batch
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=16,
                        type=int, help='batch size')
    # Sometimes we update the model after several minibatch
    parser.add_argument('--bu', '--batch-update', dest='batch_update', default=64,
                        type=int, help='batch to update')
    # Learning rate: How fast we want to train the model?
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.01,
                        type=float, help='learning rate')
    # Use GPU?
    parser.add_argument('-c', '--cpu', action='store_true', dest='cpu', default=False, help='only use cpu'),

    parser.add_argument("--sv", action="store_true", dest='save_cp', default=False, help='save model parameters'),
    parser.add_argument('-p', '--pred', action='store_true', dest='pred', default=False, help='only evaluate model')
    parser.add_argument('-l', '--load', dest='load',
                        default=False, help='load saved model parameters')
    parser.add_argument('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    parser.add_argument('-w', '--weight-decay', dest='weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--ini', '--ini-file', dest='ini_file', default='latest', type=str, help='name of the ini file')
    parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')

    # Model Parameters
    parser.add_argument('--backbone', dest='backbone', default='vgg11', type=str, help='backbone of the UNet')
    parser.add_argument('--depth', dest='depth', default=5, type=int, help='Number of layers of UNet')

    # Misc
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')

    parser.add_argument('-s', '-source', dest='source', default='imorphics',
                        type=str, help='data source to train')
    
    # tio augmentation
    parser.add_argument('--aug', nargs='*', type=str, help='tio augmentation: aff, ela, gam, noi, mot, bia, spa')
    parser.add_argument('--ap', type=float, default=1, help='augmentation probability')
    
    return parser
