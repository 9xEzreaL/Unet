import os, glob, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchio as tio
import matplotlib.pyplot as plt


def append_dict(x):
    return [j for i in x for j in i]


def resize_and_crop(pilimg, scale):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)
    pilimg = pilimg.resize((newW, newH))

    dx = 32
    w0 = pilimg.size[0]//dx * dx
    h0 = pilimg.size[1]//dx * dx
    pilimg = pilimg.crop((0, 0, w0, h0))

    return pilimg


class imorphics_masks():
    def __init__(self, adapt=None):
        self.adapt = adapt

    def load_masks(self, id, dir, fmt, scale, source):
        if self.adapt is not None:
            id = str(self.adapt.index((int(id.split('/')[1]), id.split('/')[0])) + 1) + '_' + str(int(id.split('/')[2]))
        raw_masks = []
        for d in dir:
            temp = []
            for m in d:
                x = Image.open(os.path.join(m, id + fmt))  # PIL
                x = resize_and_crop(x, scale=scale)  # PIL
                x = np.array(x)  # np.int32
                temp.append(x.astype(np.float32))  # np.float32

            raw_masks.append(temp)
        if source == 'bone_resize_B_crop_00':
            out = np.expand_dims(self.assemble_masks(raw_masks), 0) # original
        else:
            out = np.expand_dims(self.assemble_zibmasks(raw_masks), 0) # out = (1,384,384)
        return out

    # need to change float32 to long
    def assemble_zibmasks(self,raw_masks):
        converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
        for i in range(raw_masks[0][0].shape[0]):
            for j in range(raw_masks[0][0].shape[1]):
                converted_masks[i][j] += int(raw_masks[0][0][i][j])
        return converted_masks

    def assemble_masks(self, raw_masks):
        converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
        for i in range(len(raw_masks)):
            for j in range(len(raw_masks[i])):
                converted_masks[raw_masks[i][j] == 1] = i + 1
        return converted_masks


class LoaderImorphics(Dataset):
    def __init__(self, args_d, subjects_list, type):
        #  Folder of the images
        dir_img = os.path.join(args_d['data_path'], args_d['mask_name'], 'original')
        #  Folder of the masks
        dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                                  'train_masks/' + str(y) + '/') for y in x] for x in
                    args_d['mask_used']]
        self.dir_img = dir_img # original image
        self.fmt_img = glob.glob(self.dir_img+'/*')[0].split('.')[-1] # fmt = png
        self.dir_mask = dir_mask # [['/home/ziyi/Dataset/OAI_DESS_segmentation/ZIB/train_masks/png/']]
        # Assemble the masks from the folders
        self.masks = imorphics_masks(adapt=None)
        # Picking  subjects
        ids = sorted(glob.glob(self.dir_mask[0][0] + '*'))  # scan the first mask foldr
        ids = [x.split(self.dir_mask[0][0])[-1].split('.')[0] for x in ids]  # get the ids ['9001104_000...]
        self.ids = [x for x in ids if int(x.split('_')[0]) in subjects_list]  # subject name belongs to subjects_list
        # Rescale the images
        self.scale = args_d['scale']
        self.type = type
        self.source = args_d['mask_name']


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser

    def load_imgs(self, id):
        x = Image.open(self.dir_img + '/' + id + '.' + self.fmt_img).convert('L')
        x = resize_and_crop(x, self.scale)
        x = np.expand_dims(np.array(x), 0)  # to numpy
        return x

    def get_augmentation_transform(self):
        spatial_transforms = {
            tio.RandomElasticDeformation(): 0.2,
            tio.RandomAffine(): 0.8,
        }

        augment = tio.Compose([
            # tio.RandomElasticDeformation(max_displacement=3, p=0.5),
            tio.RandomAffine(),
            tio.OneOf(spatial_transforms),
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # load image
        img = self.load_imgs(id)
        # load mask
        mask = self.masks.load_masks(id, self.dir_mask, '.png', scale=self.scale, source=self.source) # (1,384,384)

        # normalization
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        img = img.type(torch.float32)
        img = img / img.max()
        img = torch.cat([1*img, 1*img, 1*img], 0) # (3,384,384)
        if self.type == 'train':
            subject = tio.Subject(
                                image=tio.ScalarImage(tensor=torch.unsqueeze(img, dim=-1)),
                                label=tio.LabelMap(tensor=torch.unsqueeze(mask, dim=-1)),
                            )
            augment = self.get_augmentation_transform()
            transformed = augment(subject)
            img = transformed.image.data.view((img.shape[0],img.shape[1],img.shape[2]))
            mask = transformed.label.data.type(torch.long).view((mask.shape[0],mask.shape[1],mask.shape[2]))

        return img, mask, id


if __name__ == '__main__':
    args_d = {'mask_name': 'bone_resize_B_crop_00',
              'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
              'mask_used': [['femur', 'tibia'], [1], [2, 3]],  # ,
              'scale': 0.5,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}

    def imorphics_split():
        train_00 = list(range(10, 71))
        eval_00 = list(range(1, 10)) + list(range(71, 89))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01

    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # datasets
    train_set = LoaderImorphics(args_d, subjects_list=train_00)

    ##
    # model
    from deep_learning.models import HEDUNet
    model = HEDUNet(input_channels=3)

    # forward
    y_hat, y_hat_levels = model(img)
    print('combined_prediction shape:')
    print(y_hat.shape)

    print('predictions list length:')
    print(len(y_hat_levels))
    for p in y_hat_levels:
        print(p.shape)

    # loss function
    from deep_learning.metrics_hedunet import HedUnetLoss
    loss_function = HedUnetLoss()

    loss = loss_function(y_hat, y_hat_levels, target)
    print(loss)