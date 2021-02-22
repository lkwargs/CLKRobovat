"""Repeated random grasp policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np

from robovat.envs.grasp import image_grasp_sampler, grasp_2d
from robovat.policies import policy
from robovat.utils.yaml_config import YamlConfig


def is_close(a, b):
    if abs(a[0] - b[0]) > 0.01:
        return True
    elif abs(a[1] - b[1]) > 0.01:
        return True
    elif abs(a[3] - b[3]) > 0.01:
        return True
    return False


class RepeatedRandomGraspPolicy(policy.Policy):
    """Antipodal grasp 4-DoF policy."""

    def __init__(self,
                 env,
                 config=None):
        """Initialize.

        Args:
            env: Environment.
            config: Policy configuration.
        """
        super(RepeatedRandomGraspPolicy, self).__init__(env, config)
        config = self.config
        self.sampler = image_grasp_sampler.AntipodalDepthImageGraspSampler(
            friction_coef=config.SAMPLER.FRICTION_COEF,
            depth_grad_thresh=config.SAMPLER.DEPTH_GRAD_THRESH,
            depth_grad_gaussian_sigma=config.SAMPLER.DEPTH_GRAD_GAUSSIAN_SIGMA,
            downsample_rate=config.SAMPLER.DOWNSAMPLE_RATE,
            max_rejection_samples=config.SAMPLER.MAX_REJECTION_SAMPLES,
            crop=config.SAMPLER.CROP,
            min_dist_from_boundary=config.SAMPLER.MIN_DIST_FROM_BOUNDARY,
            min_grasp_dist=config.SAMPLER.MIN_GRASP_DIST,
            angle_dist_weight=config.SAMPLER.ANGLE_DIST_WEIGHT,
            depth_samples_per_grasp=config.SAMPLER.DEPTH_SAMPLES_PER_GRASP,
            min_depth_offset=config.SAMPLER.MIN_DEPTH_OFFSET,
            max_depth_offset=config.SAMPLER.MAX_DEPTH_OFFSET,
            depth_sample_window_height=(
                    config.SAMPLER.DEPTH_SAMPLE_WINDOW_HEIGHT),
            depth_sample_window_width=(
                config.SAMPLER.DEPTH_SAMPLE_WINDOW_WIDTH),
            gripper_width=config.GRIPPER_WIDTH)
        self.last_action = None
        self.random_range = 0.2

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
        intrinsics = observation['intrinsics']

        grasps = self.sampler.sample(depth, intrinsics, 10)
        grasps = [grasps[np.random.randint(0, 10)]]
        grasp = np.squeeze(grasps, axis=0)
        grasp = grasp_2d.Grasp2D.from_vector(grasp, camera=self.env.camera)
        action = grasp.as_4dof()
        action[2] = -0.1

        self.last_action = action

        x = np.random.rand() * 0.2 + 0.7
        y = np.random.rand() * 0.3 - 0.15
        z = -0.1
        up_and_down = [self.last_action, [x, y, z, 0]]

        # from matplotlib import pyplot as plt
        # xl, yl, _ = np.shape(depth)
        #
        # #作图阶段
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # #定义横纵坐标的刻度
        # ax.set_yticks(range(yl))
        # ax.set_xticks(range(xl))
        # im = ax.imshow(depth)
        # plt.colorbar(im)
        # plt.show()
        
        return up_and_down
