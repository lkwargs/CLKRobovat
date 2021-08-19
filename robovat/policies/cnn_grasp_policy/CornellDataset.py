import os
import glob

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from robovat.utils.grasp_rect import GraspRectangle, rectangles2image
from imageio import imread


def normalize(img):
    minimg = np.min(img)
    maximg = np.max(img)
    return (img - minimg) * 2 / (maximg - minimg) - 1


class CornellDataset(Dataset):
    def __init__(self, data_path='/home/josep/e/orange/cornell/') -> None:
        graspf = glob.glob(os.path.join(data_path, '*', 'pcd*cpos.txt'))
        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        self.grs = []
        self.depth = []
        self.output_shape = [424, 512]
        for g, d in zip(graspf, depthf):
            grasp_rects = GraspRectangle.load_from_cornell_file(g, self.output_shape)
            self.grs.append(grasp_rects)

            depth = np.array(imread(d))
            depth = depth[:self.output_shape[0], :self.output_shape[1]]

            new_shape = (1, depth.shape[0], depth.shape[1])

            self.depth.append(depth.reshape(new_shape))
    
    def __len__(self):
        return len(self.depth)

    def __getitem__(self, idx):
        return torch.tensor(normalize(self.depth[idx]), dtype=torch.float32), \
               rectangles2image(self.grs[idx], self.depth[idx].shape)


