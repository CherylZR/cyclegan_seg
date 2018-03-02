import os.path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class UnalignedDatasetMask(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A = os.path.join(opt.dataroot, 'coco_bike')
        self.dir_MA = os.path.join(opt.dataroot, 'coco_bike_ann')
        self.dir_B = os.path.join(opt.dataroot, 'voc_bike')
        self.dir_MB = os.path.join(opt.dataroot, 'voc_bike_ann')

        self.A_paths = make_dataset(self.dir_A)
        self.MA_paths = make_dataset(self.dir_MA)
        self.B_paths = make_dataset(self.dir_B)
        self.MB_paths = make_dataset(self.dir_MB)

        self.A_paths = sorted(self.A_paths)
        self.MA_paths = sorted(self.MA_paths)
        self.B_paths = sorted(self.B_paths)
        self.MB_paths = sorted(self.MB_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.osize = [opt.loadSize, opt.loadSize]
        self.fineSize = opt.fineSize


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        MA_path = self.MA_paths[index_A]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        MB_path = self.MB_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        MA_img = Image.open(MA_path).convert('L')

        B_img = Image.open(B_path).convert('RGB')
        MB_img = Image.open(MB_path).convert('L')

        transform = transforms.Compose([
            transforms.Scale(self.osize, Image.BICUBIC),
            transforms.ToTensor()
        ])

        A_img = transform(A_img)
        A_mask = transform(A_mask)
        B_img = transform(B_img)
        B_mask = transform(B_mask)
        blob_A = torch.cat((A_img, A_mask), 0)
        blob_B = torch.cat((B_img, B_mask), 0)

        blob_A = self.transform_nc(blob_A, self.fineSize)
        blob_B = self.transform_nc(blob_B, self.fineSize)

        A = blob_A[0:3, :, :]
        MA = torch.unsqueeze(blob_A[3, :, :], 0)
        # MA = MA * 2.0 -1
        B = blob_B[0:3, :, :]
        MB = torch.unsqueeze(blob_B[3, :, :], 0)
        # MB = MB * 2.0-1
        return {'A': A, 'B': B, 'MA': MA, 'MB': MB,
                'A_paths': A_path, 'B_paths': B_path, 'MA_paths': MA_path, 'MB_paths': MB_path}

    def transform_nc(self, blob, finesize):
        [c, h, w] = blob.size()
        if random.random() < 0.5:
            blob = hflip(blob)
        i = random.randint(0, h - finesize)
        j = random.randint(0, w - finesize)
        newblob = blob[:, i:(i + finesize), j:(j + finesize)]
        return newblob

    def hflip(self, blob):
        [c, h, w] = blob.size()
        for i in range(0, h):
            for j in range(0, w // 2):
                for channel in range(0, c):
                    blob[channel, i, j], blob[channel, i, w - 1 - j] = blob[channel, i, w - 1 - j], blob[channel, i, j]
        return blob

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDatasetMask'
