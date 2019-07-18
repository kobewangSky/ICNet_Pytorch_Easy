import os
from enum import Enum
import torch
import PIL
import scipy.misc as m

from torch.utils import data
from torch import Tensor

from torchvision import transforms
import numpy as np
from Tool import recursive_glob
import random



class Cityscapesloader(data.Dataset):
    class Mode(Enum):
        TRAIN = 'train.txt'
        TEST = 'test.txt'
        VAL = 'val.txt'

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
    ]

    label_coloues = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal" : [103.939, 116.779, 123.68],
        "cityscapes" : [0.0, 0.0, 0.0],
    }

    def __init__(self, root, split = "train",  img_size=(512, 1024), augmentations = None, img_norm = True, version="cityscapes", test_mode=False):
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes  = 19
        self.img_size = img_size
        self.mean = np.array(self.mean_rgb[version])
        self.Image_files = {}
        self.label_files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.Image_files[split] = sorted(recursive_glob(rootdir= self.images_base, suffix='.png'))
        self.label_files[split] = sorted(recursive_glob(rootdir=self.annotations_base, suffix='labelIds.png'))
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.Image_files[split]:
            raise Exception("No file")

        print("Found %d %s images" % (len(self.Image_files[split]), split))

    def __len__(self):
        return len(self.Image_files[self.split])

    def __getitem__(self, item):
        img_path = self.Image_files[self.split][item].rstrip()
        lbl_path = self.label_files[self.split][item].rstrip()

        image_out = m.imread(img_path)
        label_out = m.imread(lbl_path)

        image_np = np.array(image_out)
        label_np = self.encode_segmap(np.array(label_out))

        img, lbl = self.transform(image_np, label_np)

        return img, lbl

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        #
        # lbl4 = m.imresize(lbl, (int(self.img_size[0]/4), int(self.img_size[1]/4)), "nearest", mode="F")
        # lbl8 = m.imresize(lbl, (int(self.img_size[0] / 8), int(self.img_size[1] / 8)), "nearest", mode="F")
        # lbl16 = m.imresize(lbl, (int(self.img_size[0] / 16), int(self.img_size[1] / 16)), "nearest", mode="F")
        #
        # lbl4 = lbl4.astype(int)
        # lbl8 = lbl8.astype(int)
        # lbl16 = lbl16.astype(int)




        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()

        # lbl4 = torch.from_numpy(lbl4).long()
        # lbl8 = torch.from_numpy(lbl8).long()
        # lbl16 = torch.from_numpy(lbl16).long()

        lbl = torch.from_numpy(lbl).long()

        return img, lbl














