import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from dataloaders import custom_transforms as tr

from dataloaders.transforms import Compose, RandomCrop, ToTensor

class COCO_Dataset(Dataset):

    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self, split='val', transform=None):
        super(COCO_Dataset, self).__init__()
        assert split in ['validate', 'train', 'test'], 'split argument should be train, validate or test'
        self.split = split

        #self.data_dir = 'coco/_all'
        self.transform = transform

        if self.split == 'train':
            self.data_type = 'train2017'
            self.data_dir = 'mczarnecki'
            annFile = 'mczarnecki/.exports/coco-1650865573.511341.json'
        
        if self.split == 'validate':
            self.data_type = 'train2017'
            self.data_dir = 'mczarnecki'
            annFile = 'mczarnecki/.exports/coco-1650865573.511341.json'
        
        if self.split == 'test':
            self.data_type = 'test2017'

        self.coco = COCO(annFile)

        cats = self.coco.loadCats(self.coco.getCatIds())

        self.cat_names = ['background']
        self.cat_names.extend(cat['name'] for cat in cats)

        self.cat_ids = [0]
        self.cat_ids.extend(cat['id'] for cat in cats)

        self.anno_img_id = []
        self.no_anno_img_id = []
        

        if self.split != 'test':
            for idx in self.coco.getImgIds():
                anno_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
                if len(anno_ids) == 0:
                    self.no_anno_img_id.append(idx)
                else:
                    self.anno_img_id.append(idx)
        else:
            self.anno_img_id = self.coco.getImgIds()

    def __getitem__(self, idx):
        img = self._load_img(self.anno_img_id[idx])
        w, h = img.size

        if self.split == 'test':
            seg_mask = np.zeros((w, h, 1))
        else:
            seg_mask = self._gen_seg_mask(self.anno_img_id[idx], h, w)
        
        sample = {'image': img, 'label': seg_mask}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'validate':
            return self.transform_val(sample)
            

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return len(self.anno_img_id)
    
    def _load_img(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.data_dir, img_info['file_name']))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def _gen_seg_mask(self, img_id, h, w):
        anno_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        mask_map = np.zeros((h,w), dtype=np.uint8)
        for anno in anno_info:
            rle = mask.frPyObjects(anno['segmentation'], h, w)
            m = mask.decode(rle)
            cat = anno['category_id']
            
            if cat in self.cat_ids:
                c = self.cat_ids.index(cat)
            else:
                continue

            if len(m.shape) < 3:
                mask_map[:, :] += (mask_map == 0) * (m * c)
            else:
                mask_map[:, :] += (mask_map == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)

        return Image.fromarray(mask_map)