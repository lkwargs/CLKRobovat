import os
import time
import pickle
import cv2 as cv
import csv
import numpy as np
from collections import defaultdict


class Saver(object):

    def __init__(self, save_path):
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.save_path = save_path + localtime
        os.mkdir(self.save_path)
        self.num_image_saved = defaultdict(lambda: -1)


    def save(self, image, action, obj_name):
        if self.num_image_saved[obj_name] == -1:
            os.mkdir(self.save_path + "/" + obj_name)
            self.num_image_saved[obj_name] = 0
        filename = self.save_path + "/" + obj_name + "/" + \
                   str(self.num_image_saved[obj_name]) + ".txt"
        data = {"depth": image["depth"], "rgb": image["rgb"], "segmask": image["segmask"]}
        with open(filename, "wb") as f:
            pickle.dump([data, action], f)
            f.close()

        self.num_image_saved[obj_name] += 1



def write2csv(filename, rows):
    with open(filename, "w", encoding='utf-8', newline='') as f:
        wt = csv.writer(f)
        wt.writerows(rows)
        f.close()


class SingleObjSaver(object):

    def __init__(self, save_path, poses_per_obj, grasps_per_pose):
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.save_path = save_path + '/' + localtime
        os.mkdir(self.save_path)

        self.poses_per_obj = poses_per_obj
        self.grasps_per_pose = grasps_per_pose
        
        self.current_pixel_grasps = []
        self.current_xyz_grasps = []
        self.current_pose = None
        self.num_pose_saved = defaultdict(lambda: -1)

    def save(self, image, action, obj_name):
        if self.num_pose_saved[obj_name] == -1:
            os.mkdir(self.save_path + "/" + obj_name)
            self.num_pose_saved[obj_name] = 0

        if len(self.current_pixel_grasps) == 0:
            os.mkdir(self.save_path + "/" + obj_name + "/" + str(self.num_pose_saved[obj_name]))
            self.current_pose = image
        
        self.current_pixel_grasps.append(action['pixel'])
        self.current_xyz_grasps.append(action['xyz'])

        if len(self.current_pixel_grasps) == self.grasps_per_pose:
            current_path = self.save_path + "/" + obj_name + "/" + \
                            str(self.num_pose_saved[obj_name]) + "/"

            depth_name = current_path + "depth.npy"
            rgb_name = current_path + "rgb.png"
            seg_name = current_path + "seg.npy"

            pixel_name = current_path + "pixel.csv"
            xyz_name = current_path + "xyz.csv"

            # write to files
            cv.imwrite(rgb_name, self.current_pose['rgb'])
            np.save(depth_name, self.current_pose['depth'])
            np.save(seg_name, self.current_pose['segmask'])

            write2csv(pixel_name, self.current_pixel_grasps)
            write2csv(xyz_name, self.current_xyz_grasps)

            # reset
            self.current_pixel_grasps = []
            self.current_xyz_grasps = []
            self.current_pose = None

            # count
            self.num_pose_saved[obj_name] += 1

    @staticmethod
    def load(pose_dir):
        csv_path = pose_dir + "/pixel.csv"
        depth_path = pose_dir + "/depth.npy"

        with open(csv_path, "r", encoding='utf-8', newline='') as f:
            rd = csv.reader(f)
            grasps = [np.array(g, dtype=np.float) for g in rd]

        depth = np.load(depth_path)

        return grasps, depth