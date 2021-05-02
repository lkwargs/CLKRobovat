"""This is an example environment for grasping given a specific pose. This environment assumes one object in the environment only.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random
import os.path
import cv2

import gym
import numpy as np
import time 

from robovat.envs import arm_env
from robovat.utils.grasp_2d import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.observations import camera_obs
from robovat.reward_fns.grasp_reward import GraspReward
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig

GRASPABLE_NAME = 'graspable'

class FrankaPandaGraspEnv(arm_env.ArmEnv):
    """An example franka panda grasping environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        """Initialize.

        Args:
            simulator: Instance of the simulator.
            config: Environment configuration.
            debug: True if it is debugging mode, False otherwise.
        """
        self._simulator = simulator
        self._config = config or self.default_config
        self._debug = debug

        # Camera.
        self.camera = self._create_camera(
            height=self.config.KINECT2.DEPTH.HEIGHT,
            width=self.config.KINECT2.DEPTH.WIDTH,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION)

        # Graspable object.
        if self.is_simulation:
            self.graspable = None
            self.graspable_path = None
            self.graspable_pose = None
            self.all_graspable_paths = []
            self.graspable_index = 0

            for pattern in self.config.SIM.GRASPABLE.PATHS:
                if not os.path.isabs(pattern):
                    pattern = os.path.join(self.simulator.assets_dir, pattern)

                if pattern[-4:] == '.txt':
                    with open(pattern, 'r') as f:
                        paths = [line.rstrip('\n') for line in f]
                else:
                    paths = glob.glob(pattern)

                print(pattern)

                self.all_graspable_paths += paths

            self.all_graspable_paths.sort()
            num_graspable_paths = len(self.all_graspable_paths)
            assert num_graspable_paths > 0, (
                'Found no graspable objects at %s' % (self.config.SIM.GRASPABLE.PATHS))
            logger.debug('Found %d graspable objects.', num_graspable_paths)

        super(FrankaPandaGraspEnv, self).__init__(
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

    @property
    def default_config(self):
        """Load the default configuration file."""
        config_path = os.path.join('configs', 'envs', 'grasp_4dof_env.yaml')
        assert os.path.exists(config_path), (
                'Default configuration file %s does not exist' % (config_path))
        return YamlConfig(config_path).as_easydict()

    def _create_observations(self):
        """Create observations.

        Returns:
            List of observations.
        """
        return [
            camera_obs.CameraObs(
                name=self.config.OBSERVATION.TYPE,
                camera=self.camera,
                modality=self.config.OBSERVATION.TYPE,
                max_visible_distance_m=None),
            camera_obs.CameraIntrinsicsObs(
                name='intrinsics',
                camera=self.camera),
            camera_obs.CameraTranslationObs(
                name='translation',
                camera=self.camera),
            camera_obs.CameraRotationObs(
                name='rotation',
                camera=self.camera)
        ]

    def _create_reward_fns(self):
        """Initialize reward functions.

        Returns:
            List of reward functions.
        """
        if self.simulator is None:
            raise NotImplementedError(
                'Need to implement the real-world grasping reward.'
            )

        return [
            GraspReward(
                name='grasp_reward',
                end_effector_name=self.config.SIM.ARM.ARM_NAME,
                graspable_name=GRASPABLE_NAME)
        ]

    def step(self, action):
        """Take a step for affordence exploration.
        """
        # if self._done:
        #     raise ValueError('The environment is done. Forget to reset?')
       
        self._execute_action(action)

        self._num_steps += 1

        observation = self.get_observation()

        reward, termination = self.get_reward()

        self._episode_reward += reward
        self._done = (self._done or termination)

        if self.config.MAX_STEPS is not None:
            if self.num_steps >= self.config.MAX_STEPS:
                self._done = True

        logger.info('step: %d, reward: %.3f', self.num_steps, reward)

        if self._done:
            self._num_episodes += 1
            self._total_reward += self.episode_reward
            logger.info(
                'episode_reward: %.3f, avg_episode_reward: %.3f',
                self.episode_reward,
                float(self.total_reward) / (self._num_episodes + 1e-14),
            )

        if self.debug:
            self.render()

        return observation, reward, self._done, None

    def _create_action_space(self):
        """Create the action space.

        Returns:
            The action space.
        """
        if self.config.ACTION.TYPE == 'CUBOID':
            low = self.config.ACTION.CUBOID.LOW + [0.0]
            high = self.config.ACTION.CUBOID.HIGH + [2 * np.pi]
            return gym.spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    dtype=np.float32)
        elif self.config.ACTION.TYPE == 'IMAGE':
            height = self.camera.height
            width = self.camera.width
            return gym.spaces.Box(
                low=np.array([0, 0, 0, 0, -(2*24 - 1)]),
                high=np.array([width, height, width, height, 2*24 - 1]),
                dtype=np.float32)
        else:
            raise ValueError

    def _reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        print("reset scene....")
        # Configure debug visualizer
        if self._debug and self.is_simulation:
            visualizer_info = self.simulator.physics.get_debug_visualizer_info(['yaw', 'pitch', 'dist', 'target'])
            visualizer_info['dist'] = 1.5
            visualizer_info['yaw'] = 90
            visualizer_info['pitch'] = -20
            self.simulator.physics.reset_debug_visualizer(camera_distance=visualizer_info['dist'],
                                                         camera_yaw=visualizer_info['yaw'],
                                                         camera_pitch=visualizer_info['pitch'],
                                                         camera_target_position=visualizer_info['target'])
  
        super(FrankaPandaGraspEnv, self)._reset_scene()

        # load all the graspable objects
        for object_path in self.all_graspable_paths[5:10]:
            pose = Pose.uniform(x=self.config.SIM.GRASPABLE.POSE.X,
                    y=self.config.SIM.GRASPABLE.POSE.Y,
                    z=self.config.SIM.GRASPABLE.POSE.Z,
                    roll=self.config.SIM.GRASPABLE.POSE.ROLL,
                    pitch=self.config.SIM.GRASPABLE.POSE.PITCH,
                    yaw=self.config.SIM.GRASPABLE.POSE.YAW)
            pose = get_transform(source=self.table_pose).transform(pose)
            scale = np.random.uniform(*self.config.SIM.GRASPABLE.SCALE)
            
            logger.info('Loaded the graspable object from %s with scale %.2f...', self.graspable_path, scale)
            self.graspable = self.simulator.add_body(object_path, \
                                                     pose, 
                                                     scale=scale, 
                                                     name=GRASPABLE_NAME) # TODO: name
            logger.debug('Waiting for graspable objects to be stable...')
            self.simulator.wait_until_stable(self.graspable)
        
        # Change some properties of the object compensating for artifacts due to simulation
        self.table.set_dynamics(contact_damping=100., contact_stiffness=100.)    
                
        # Reset camera.
        self._reset_camera(
            self.camera,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION,
            intrinsics_noise=self.config.KINECT2.DEPTH.INTRINSICS_NOISE,
            translation_noise=self.config.KINECT2.DEPTH.TRANSLATION_NOISE,
            rotation_noise=self.config.KINECT2.DEPTH.ROTATION_NOISE)

    def _reset_robot(self):
        """Reset the robot in simulation or the real world.
        """
        super(FrankaPandaGraspEnv, self)._reset_robot()
        self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)
        
    def _execute_action(self, action):
        """Execute the grasp action.

        Args:
            action: A 4-DoF grasp defined in the image space or the 3D space.
        """
        if self.config.ACTION.TYPE == 'CUBOID':
            x, y, z, angle = action
        elif self.config.ACTION.TYPE == 'IMAGE':
            grasp = Grasp2D.from_vector(action, camera=self.camera)
            x, y, z, angle = grasp.as_4dof()
        else:
            raise ValueError(
                'Unrecognized action type: %r' % (self.config.ACTION.TYPE))

                
        print("Grasp 4DoF pose: ", x, y, z, angle)
        start = Pose(
            [[x, y, z + self.config.ARM.FINGER_TIP_OFFSET], [0, np.pi, angle]]
        )
        put_pose = start.copy()

        #################################################
        # visualize current image frame.
        
        images = self.camera.frames()
        from IPython import embed; embed()

        image = images['rgb']
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', image)

        #################################################
        # state by 'initial' phase.

        phase = 'initial'
        # Handle the simulation robustness.
        if self.is_simulation:
            num_action_steps = 0

        while phase != 'done':

            if self.is_simulation:
                self.simulator.step()
                if phase == 'start':
                    num_action_steps += 1

            if self._is_phase_ready(phase, num_action_steps):
                
                phase = self._get_next_phase(phase)
                logger.debug('phase: %s', phase)
                if phase == 'overhead':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)

                elif phase == 'prestart':
                    prestart = start.copy()
                    prestart.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(prestart)

                elif phase == 'start':
                    self.robot.move_to_gripper_pose(start, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.is_simulation:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.table.set_dynamics(
                            lateral_friction=100)

                elif phase == 'end':
                    self.robot.grip(1)

                elif phase == 'pickup':
                    pickup = self.robot.end_effector.pose
                    pickup.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(
                        pickup, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.is_simulation:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.table.set_dynamics(
                            lateral_friction=1)

                elif phase == 'release':
                    self.robot.grip(0)

                elif phase == 'reset':
                    self.robot.move_to_joint_positions(self.config.ARM.OFFSTAGE_POSITIONS)
                    # pickup = self.robot.end_effector.pose
                    # pickup.z = 0.6
                    # self.robot.move_to_gripper_pose(
                    #     pickup, straight_line=True)

    def _is_phase_ready(self, phase, num_action_steps):
        """Check if the current phase is ready.

        Args:
            phase: A string variable.
            num_action_steps: Number of steps in the `start` phase.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.is_simulation:
            if phase == 'start':
                if num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    logger.debug('The grasping motion is stuck.')
                    return True

            if phase == 'start' or phase == 'end':
                if self.simulator.check_contact(self.robot.arm, self.table):
                    logger.debug('The gripper contacts the table')
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            return True
        else:
            return False

    def _get_next_phase(self, phase):  # TODO: add the push phase here
        """Get the next phase of the current phase.  
        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.
        """
        phase_list = ['initial',
                      'overhead',
                      'prestart',
                      'start',
                      'end',
                      'pickup',
                      'release',
                      'reset',
                      'done']
        
        i = phase_list.index(phase)
        if i == len(phase_list):
            raise ValueError('phase %r does not have a next phase.')
        else:
            return phase_list[i + 1]
        raise ValueError('Unrecognized phase: %r' % phase)
