import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tools.sample_saver import SingleObjSaver
from robovat.utils.grasp_rect import GraspRectangle, rectangles2image


def normalize(img):
    minimg = np.min(img)
    maximg = np.max(img)
    return (img - minimg) * 2 / (maximg - minimg) - 1


class RobovatDataset(Dataset):
    def __init__(self, data_path='/home/josep/e/samples/2021-07-09 20:24:03/wooden_cup1/') -> None:
        pose_dirs = os.listdir(data_path)
        self.grs = []
        self.depth = []
        for d in pose_dirs:
            pose_dir = data_path + d
            grasp_2ds, depth = SingleObjSaver.load(pose_dir)
            grasp_rects = [GraspRectangle.from_grasp2d(g2d[0:2], g2d[3]) for g2d in grasp_2ds]
            self.grs.append(grasp_rects)

            new_shape = (1, depth.shape[0], depth.shape[1])

            self.depth.append(depth.reshape(new_shape))
    
    def __len__(self):
        return len(self.depth)

    def __getitem__(self, idx):
        return torch.tensor(normalize(self.depth[idx]), dtype=torch.float32), \
               rectangles2image(self.grs[idx], self.depth[idx].shape)


if __name__ == '__main__':
    dataset = RobovatDataset()
    import matplotlib.pyplot as plt
    for i in range(dataset.__len__()):
        x, y = dataset.__getitem__(i)
        x = x[0]
        plt.subplot(231)
        plt.imshow(x)
        for j in range(len(y)):
            plt.subplot(231 + j + 1)
            plt.imshow(y[j][0])
        plt.show()
