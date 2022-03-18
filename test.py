import torch
import os
import glob
from utils.imagesc import imagesc
from utils.args import args_train, merge_args
from dotenv import load_dotenv
load_dotenv('.env')
from loaders.loader_imorphics import LoaderImorphics as Loader


# args_d = {'mask_name': 'bone_resize_B_crop_00',
#           'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
#           'mask_used': [['femur'], ['tibia']],  # [[1], [2, 3]],  # ['femur'], ['tibia'],
#           'scale': 0.5,
#           'interval': 1,
#           'thickness': 0,
#           'method': 'automatic'}


def imorphics_split():
    train_00 = list(range(10, 71))
    eval_00 = list(range(1, 10)) + list(range(71, 89))
    train_01 = list(range(10 + 88, 71 + 88))
    eval_01 = list(range(1 + 88, 10 + 88)) + list(range(71 + 88, 89 + 88))
    return train_00, eval_00, train_01, eval_01

def zib_split():
    dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                              'train_masks/' + str(y) + '/') for y in x] for x in
                args_d['mask_used']]
    ids = sorted(glob.glob(dir_mask[0][0] + '*'))  # scan the first mask foldr
    ids = [x.split(dir_mask[0][0])[-1].split('.')[0] for x in ids]  # get the ids ['9001104_000...]
    ids = [int(x.split('_')[0]) for x in ids]
    ids = list(set(ids))

    train_00 = list(ids[5:20])
    eval_00 = list(ids[:5]) + list(ids[400:404])
    return train_00, eval_00


def simple_test():
    """
    A simple test function to prnt out
    """
    train_00, eval_00= zib_split()

    # Dataloader
    # train_set = Loader(args, subjects_list=train_00, type='train')
    eval_set = Loader(args, subjects_list=eval_00, type='eval')

    # Loading Data
    x, y, id = eval_set.__getitem__(50)
    print(id)
    print('shape of input')
    print(x.shape)
    print('shape of label')
    print(y.shape)

    #
    model = torch.load('largecheckpoints/model_seg.pth')
    out, = model(x.unsqueeze(0).cuda())
    print(out.shape)
    segmentation = torch.argmax(out, 1).detach().cpu()
    print(segmentation.shape)

    to_print = torch.cat([x/x.max() for x in [x[0, ::], segmentation[0, ::]]], 1)
    imagesc(to_print, show=False, save='segmentation.png')


if __name__ == '__main__':
    # Training Arguments
    parser = args_train()
    args = dict(vars(parser.parse_args()))
    args['dir_checkpoint'] = os.environ.get('CHECKPOINTS')
    args_d = {'mask_name': 'ZIB',
              'data_path': '/home/ziyi/Dataset/OAI_DESS_segmentation',
              #                   os.getenv("HOME") + os.environ.get('DATASET'),
              'mask_used': [['png']],  # [[1], [2, 3]],  # ['femur'], ['tibia'],
              'scale': 1,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}
    args = merge_args(args, args_d)

    simple_test()