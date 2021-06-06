from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pickle

import _init_paths
# sys.path.append("/home/josep/code/python/rlcode/robovat")
from obs_handler import ObservationHandler


class SampleVisualizer(object):

    def __init__(self, samples_dir):
        if not os.path.exists(samples_dir):
            print("Sample directory does not exist.")
            exit(-1)
        elif not os.path.isdir(samples_dir):
            print(samples_dir + " is not a directory.")
            exit(-1)

        dirs = [samples_dir + '/' + _ for _ in os.listdir(samples_dir)]
        self.files = []
        for d in dirs:
            self.files.extend([d + '/' + _ for _ in os.listdir(d)])

        self.show_idx = 0
        self.plotter = ObservationHandler()

    def show_next(self):
        with open(self.files[self.show_idx], 'rb') as f:
            data = pickle.load(f)
            observation, pick_place = data[0], data[1]
            grasp = pick_place[0]
            self.plotter.plot([observation['depth'],
                               observation['rgb'],
                               observation['segmask']],
                              grasp
                              )
            self.plotter.show()
            self.show_idx += 1
            f.close()

    def show_all(self):
        while self.show_idx < len(self.files):
            self.show_next()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, help="Directory of samples.")
    args = parser.parse_args()

    v = SampleVisualizer(args.sample_dir)
    v.show_all()
