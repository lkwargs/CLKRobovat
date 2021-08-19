"""CNN grasp policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import torch

from torch.utils.data import DataLoader
from robovat.policies.cnn_grasp_policy.ggcnn import GGCNN2
from robovat.policies.cnn_grasp_policy.RobovatDataset import RobovatDataset
from robovat.policies.cnn_grasp_policy.CornellDataset import CornellDataset
from robovat.policies.cnn_grasp_policy.RobovatDataset import normalize
from robovat.policies import policy
from robovat.utils.grasp_2d import Grasp2D
from robovat.utils.yaml_config import YamlConfig
from tools.obs_handler import ObservationHandler

device = torch.device('cuda')


class CNNGraspPolicy(policy.Policy):

    """CNN grasp 4-DoF policy."""

    def __init__(self, env, config=None):
        """Initialize.

        Args:
            env: Environment.
            config: Policy configuration.
        """
        super(CNNGraspPolicy, self).__init__(env, config)
        self.agent = GGCNN2().to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters())
        self.plotter = ObservationHandler()
        if self.config.TRAIN:
            self.pretrain()
            self.train()
        else:
            self.load('/home/josep/code/python/rlcode/robovat/results/2021-07-25 21:23:13')

    def pretrain(self, data_path=None, epochs=10):
        if data_path is None:
            dataset = CornellDataset()
        else:
            dataset = CornellDataset(data_path)

        train_data = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
        for epoch in range(epochs):
            for x, y in train_data:

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = self.agent.compute_loss(xc, yc)

                loss = lossd['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print('Epoch: {}, Loss: {:0.4f}'.format(epoch, loss.item()))
        
    def train(self, data_path=None, epochs=20):
        if data_path is None:
            dataset = RobovatDataset()
        else:
            dataset = RobovatDataset(data_path)
        train_data = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
        for epoch in range(epochs):
            for x, y in train_data:

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = self.agent.compute_loss(xc, yc)
                # import matplotlib.pyplot as plt
                # plt.imshow(lossd['pred']['pos'][0].cpu().detach().numpy().squeeze())
                # plt.show()

                loss = lossd['loss']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print('Epoch: {}, Loss: {:0.4f}'.format(epoch, loss.item()))
        
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.save("results/" + localtime)

    @property
    def default_config(self):
        """Load the default configuration file."""
        config_path = os.path.join('configs', 'policies',
                                   'repeated_random_grasp_policy.yaml')
        assert os.path.exists(config_path), (
                'Default configuration file %s does not exist' % (config_path)
        )
        return YamlConfig(config_path).as_easydict()

    def _action(self, observation):
        """Implementation of action.

        Args:
            observation: The observation of the current step.

        Returns:
            action: The action of the current step.
        """

        depth = observation['depth']
        depth = normalize(depth)

        new_shape = (1, 1, depth.shape[0], depth.shape[1])
        net_input = torch.tensor(depth.reshape(new_shape), dtype=torch.float32, device=device)

        pos, cos, sin, width = self.agent(net_input)
        pos, cos, sin, width = pos.cpu().detach().numpy().squeeze(), \
                               cos.cpu().detach().numpy().squeeze(), \
                               sin.cpu().detach().numpy().squeeze(), \
                               width.cpu().detach().numpy().squeeze()

        center = np.unravel_index(np.argmax(pos), pos.shape)

        tan = sin[center[0]][center[1]] / cos[center[0]][center[1]]
        dph = depth[center[0]][center[1]][0]
        center = [center[1], center[0]]

        print(center, dph, tan)

        import matplotlib.pyplot as plt
        plt.subplot(231)
        plt.imshow(depth)
        y = [pos, cos, sin, width]
        for j in range(len(y)):
            plt.subplot(231 + j + 1)
            plt.imshow(y[j])
        plt.show()

        grasp = Grasp2D(center, tan, dph, 0.05, self.env.camera)
        place = [np.random.rand() * 0.2 + 0.7, np.random.rand() * 0.3 - 0.15, 0, 0]
        pick_place = [grasp, place]

        return pick_place

    def save(self, filename):
        torch.save(self.agent.state_dict(), filename)

    def load(self, filename):
        self.agent.load_state_dict(torch.load(filename))

