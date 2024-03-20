import glob
import os
import pickle
import random

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import DataLoader

def to_int(x):
    return tuple(map(int, x))

class AuxDataset(torch.utils.data.Dataset):
    def __init__(self, flist, batch_size,
                 mask_path=None, add_pos=False,
                 train_line_path=None, eval_line_path=None, wireframe_th=0.85,
                 augment=True, training=True, test_mask_path=None,
                 input_size=None, load_path=False, default_size=256, str_size=256,
                 world_size=1, mask_rate=[0.4, 0.8, 1.0], min_sigma=2.0, max_sigma=2.5):
        super(AuxDataset, self).__init__()
        self.augment = augment
        self.training = training
        self.train_line_path = train_line_path
        self.eval_line_path = eval_line_path
        self.wireframe_th = wireframe_th
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        self.data = []
        if load_path:
            self.data = glob.glob(flist + '/*')
            self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        else:
            f = open(flist, 'r')
            for i in f.readlines():
                i = i.strip()
                self.data.append(i)
            f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.default_size = default_size
        if input_size is None:
            self.input_size = default_size
        else:
            self.input_size = input_size
        self.str_size = str_size  # 256 fortransformer
        self.world_size = world_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        img_256 = self.resize(img, self.str_size, self.str_size)

        # load mask
        mask = self.load_mask(img, index)
        mask_256 = cv2.resize(mask, (self.str_size, self.str_size), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255

        gray = rgb2gray(img)
        edge = self.load_edge(gray, sigma=self.min_sigma + ((size - 256) / 256 * (self.max_sigma - self.min_sigma)))
        gray_256 = rgb2gray(img_256)
        edge_256 = self.load_edge(gray_256, sigma=self.min_sigma + (
                (self.str_size - 256) / 256 * (self.max_sigma - self.min_sigma)))

        # load line
        line = self.load_wireframe(index, size)
        line_256 = self.load_wireframe(index, self.str_size)

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
            img_256 = img_256[:, ::-1, ...].copy()
            edge = edge[:, ::-1, ...].copy()
            edge_256 = edge_256[:, ::-1, ...].copy()
            line = line[:, ::-1, ...].copy()
            line_256 = line_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
            mask_256 = mask_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()
            mask_256 = mask_256[::-1, :, ...].copy()

        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['img_256'] = self.to_tensor(img_256, norm=True)
        batch['mask'] = self.to_tensor(mask)
        batch['mask_256'] = self.to_tensor(mask_256)
        batch['edge'] = self.to_tensor(edge)
        batch['edge_256'] = self.to_tensor(edge_256)
        batch['line'] = self.to_tensor(line)
        batch['line_256'] = self.to_tensor(line_256)
        batch['size_ratio'] = size / self.default_size

        batch['name'] = self.load_name(index)
        return batch

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rate[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rate[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def load_edge(self, img, sigma=2):
        return canny(img, sigma=sigma, mask=None).astype(np.float)

    def load_wireframe(self, idx, size):
        selected_img_name = self.data[idx]
        line_name = os.path.basename(selected_img_name)
        if self.training is False:
            line_name = self.eval_line_path + '/' + line_name
        else:
            line_name = self.train_line_path + '/' + line_name
        line_name = line_name.replace('.png', '.pkl').replace('.jpg', '.pkl')

        wf = pickle.load(open(line_name, 'rb'))
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):
            if wf['scores'][i] > self.wireframe_th:
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
        return lmap
