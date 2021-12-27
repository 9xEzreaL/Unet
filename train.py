import time, os
import glob
from torch.utils.data import DataLoader
from utils.args import args_train, merge_args
import torch.nn as nn
from engine.lightning_classification import LitClassification
from dotenv import load_dotenv
load_dotenv('.env')
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


def train(net, args, train_set, eval_set, loss_function, metrics):
    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'], shuffle=False, num_workers=4, drop_last=False)

    # Load the model to GPU
    if args['legacy']:
        if not args['cpu']:
            net = net.cuda()
        # Run the models in parallel
        if args['parallel']:
            net = nn.DataParallel(net)

    # Sometimes with freeze a part of the model to reduce the number of parameters
    net.par_freeze = []

    # Define Final Model
    ln_classification = LitClassification(args=args,
                                          train_loader=train_loader,
                                          eval_loader=eval_loader,
                                          net=net,
                                          loss_function=loss_function,
                                          metrics=metrics)
    if args['legacy']:
        # Use pytorch without lightning
        ln_classification.overall_loop()
    else:
        # Use pytorch lightning for training, you can ignore it
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/' + args['prj'] + '/',
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        # we can use loggers (from TensorBoard) to monitor the progress of training
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args['prj'] + '/')
        trainer = pl.Trainer(gpus=1, accelerator='ddp',
                             max_epochs=100, progress_bar_refresh_rate=20, logger=tb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(ln_classification, train_loader, eval_loader)


if __name__ == "__main__":
    from loaders.loader_imorphics import LoaderImorphics as Loader
    from models.unet import UNet_clean
    from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient

    # Training Arguments
    parser = args_train()
    args = dict(vars(parser.parse_args()))
    args['dir_checkpoint'] = os.environ.get('CHECKPOINTS')

    # Splitting Subjects
    def imorphics_split():
        train_00 = list(range(10, 71))
        eval_00 = list(range(1, 10)) + list(range(71, 89))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01


    def zib_split():
        dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                                  'train_masks/' + str(y) + '/') for y in x] for x in
                    args_d['mask_used']]
        ids = sorted(glob.glob(dir_mask[0][0] + '*'))  # scan the first mask foldr
        ids = [x.split(dir_mask[0][0])[-1].split('.')[0] for x in ids]  # get the ids ['9001104_000...]
        ids = [int(x.split('_')[0]) for x in ids]
        ids = list(set(ids))

        train_00 = list(ids[5:400])
        eval_00 = list(ids[:5]) + list(ids[400:404])
        return train_00, eval_00

    # Datasets
    # Dataset Arguments
    if args['source'] == 'imorphics':
        args_d = {'mask_name': 'bone_resize_B_crop_00',
                  'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
                  'mask_used': [[1],[2, 3]],  #[[1], [2, 3]],  # ['femur'], ['tibia'],
                  'scale': 0.5,
                  'interval': 1,
                  'thickness': 0,
                  'method': 'automatic'}
        args_l = {'classes': len(args_d['mask_used'])+1 }
        args_d = merge_args(args_d, args_l)
        train_00, eval_00, train_01, eval_01 = imorphics_split()

    else:
        args_d = {'mask_name': 'ZIB',
                  'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
                  'mask_used': [['png']],  #[[1], [2, 3]],  # ['femur'], ['tibia'],
                  'scale': 1,
                  'interval': 1,
                  'thickness': 0,
                  'method': 'automatic',
                  'classes': 5}
        train_00, eval_00 = zib_split()
    args = merge_args(args, args_d)

    # Dataloader
    train_set = Loader(args_d, subjects_list=train_00, type='train')
    eval_set = Loader(args_d, subjects_list=eval_00, type='eval')
    print('Length of training set')
    print(len(train_set))
    print('Length of Validation set')
    print(len(eval_set))

    # new API
    from segmentation_models_pytorch.unet import Unet
    net = Unet(encoder_name=args['backbone'], classes=args['classes'], activation=None, encoder_depth=5)

    # Model, Loss Function, Metrics
    # old model API
    # net = UNet_clean(output_ch = 5, backbone=args['backbone'], depth=args['depth'])
    loss_function = SegmentationCrossEntropyLoss()
    metrics = SegmentationDiceCoefficient()

    # Start Training
    train(net, args, train_set, eval_set, loss_function, metrics)

# Usage in command line:
# CUDA_VISIBLE_DEVICES=0 python train.py -b 32 --bu 64 --lr 0.01 -s imorphics --legacy