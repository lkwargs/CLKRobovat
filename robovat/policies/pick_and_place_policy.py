"""Repeated random grasp policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np

from robovat.policies.segment_mask_sampler import SegmentationGraspSampler
from robovat.policies.image_grasp_sampler import AntipodalDepthImageGraspSampler
from robovat.policies import policy
from tools.obs_handler import ObservationHandler

from robovat.utils.yaml_config import YamlConfig
from robovat.utils.grasp_2d import Grasp2D
from robovat.utils import visualize


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
        if config.USE_HEURISTIC_POLICY:
            self.sampler = AntipodalDepthImageGraspSampler(
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

        else:
            self.sampler = SegmentationGraspSampler(config.GRIPPER_WIDTH)

        self.last_action = None
        self.random_range = 0.2
        self.plotter = ObservationHandler()

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
        if self.config.USE_HEURISTIC_POLICY:
            depth = observation['depth']
            intrinsics = observation['intrinsics']

            grasps = self.sampler.sample(depth, intrinsics, self.config.NUM_SAMPLES)
            grasp = np.squeeze(grasps, axis=0)

            grasp = Grasp2D.from_vector(grasp, camera=self.env.camera)
            visualize.plot_grasp_on_image(depth, grasp)

        else:
            depth = observation['depth']
            segmask = observation['segmask']
            intrinsics = observation['intrinsics']

            grasps = self.sampler.sample(segmask, depth, intrinsics, self.config.NUM_SAMPLES)
            grasp = np.squeeze(grasps, axis=0)
            center = [grasp[0], grasp[1]]
            angle = grasp[2]
            grasp = Grasp2D(center, angle, grasp[3], grasp[4], self.env.camera)

        if self.config.SHOW_IMAGE:
            self.plotter.plot([observation['depth'],
                               observation['rgb'],
                               observation['segmask']],
                              grasp
                              )
            self.plotter.show()

        place = [np.random.rand() * 0.2 + 0.7, np.random.rand() * 0.3 - 0.15, 0, 0]
        pick_place = [grasp, place]

        return pick_place
