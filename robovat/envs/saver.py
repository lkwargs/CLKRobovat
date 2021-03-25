import os
import time
import pickle
from collections import defaultdict


class Saver(object):

    def __init__(self):
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.save_path = "/home/josep/code/python/rlcode/robovat/samples/" + localtime
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
