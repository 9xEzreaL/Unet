import os, glob, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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

    def load_masks(self, id, dir, fmt, scale):
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

        out = np.expand_dims(self.assemble_masks(raw_masks), 0)
        return out

    def assemble_masks(self, raw_masks):
        converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
        for i in range(len(raw_masks)):
            for j in range(len(raw_masks[i])):
                converted_masks[raw_masks[i][j] == 1] = i + 1

        return converted_masks


class LoaderImorphics(Dataset):
    def __init__(self, args_d, subjects_list):
        #  Folder of the images
        dir_img = os.path.join(args_d['data_path'], args_d['mask_name'], 'original')
        #  Folder of the masks
        dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                                  'train_masks/' + str(y) + '/') for y in x] for x in
                    args_d['mask_used']]

        self.dir_img = dir_img # /home/ziyi/Dataset/OAI_DESS_segmentation/ZIB/original/
        self.fmt_img = glob.glob(self.dir_img+'/*')[0].split('.')[-1] # png
        self.dir_mask = dir_mask # [['/home/ziyi/Dataset/OAI_DESS_segmentation/ZIB/train_masks/png/']]
        # Assemble the masks from the folders
        self.masks = imorphics_masks(adapt=None)

        # Picking  subjects
        ids = sorted(glob.glob(self.dir_mask[0][0] + '*'))  # scan the first mask foldr
        ids = [x.split(self.dir_mask[0][0])[-1].split('.')[0] for x in ids]  # get the ids
        self.ids = [x for x in ids if int(x.split('_')[0]) in subjects_list]  # subject name belongs to subjects_list

        # Rescale the images
        self.scale = args_d['scale']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--encoder_layers", type=int, default=12)
        parser.add_argument("--data_path", type=str, default="/some/path")
        return parent_parser

    def load_imgs(self, id):
        x = Image.open(self.dir_img + id + '.' + self.fmt_img)
        x = resize_and_crop(x, self.scale)
        x = np.expand_dims(np.array(x), 0)  # to numpy
        return x

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # load image
        img = self.load_imgs(id)

        # load mask
        mask = self.masks.load_masks(id, self.dir_mask, '.png', scale=self.scale)

        # normalization
        img = torch.from_numpy(img)
        img = img.type(torch.float32)
        img = img / img.max()

        img = torch.cat([1*img, 1*img, 1*img], 0)

        return img, mask, id

