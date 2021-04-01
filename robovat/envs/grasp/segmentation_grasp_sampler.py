import numpy as np

from robovat.perception.camera.camera import Camera


class SegmentationGraspSampler(object):
    """Segmentation image grasp sampler.

    Use segmentation image to sample grasp actions.
    """

    def __init__(self, gripper_width):
        self.gripper_width = gripper_width


    def sample(self, image, camera, num_samples):

        if not isinstance(camera, Camera):
            intrinsics = camera
            camera = Camera()
            camera.set_calibration(intrinsics, np.zeros((3,)), np.zeros((3,)))

        image = np.squeeze(image, -1)
        obj_exists = np.max(image)
        obj_selected = np.random.randint(2, obj_exists + 1)

        # Sample grasp point
        valid_points = np.where(image == obj_selected)
        valid_idx = np.random.choice(valid_points[0].shape[0], num_samples)
        center_samples = np.c_[valid_points[1][valid_idx], valid_points[0][valid_idx]]
        angle_samples = (np.random.random(num_samples) * 2 - 1) * np.pi
        max_depth = 0.825 + 0.015
        depth_samples = np.array([max_depth] * num_samples)

        if self.gripper_width > 0:
            p1 = np.array([0, 0, max_depth])
            p2 = np.array([self.gripper_width, 0, max_depth])
            u1 = camera.project_point(p1, is_world_frame=False)
            u2 = camera.project_point(p2, is_world_frame=False)
            max_grasp_width_pixel = np.linalg.norm(u1 - u2)
        else:
            max_grasp_width_pixel = np.inf
        width_samples = np.array([0.05] * num_samples)
        actions = np.c_[center_samples,
                        angle_samples,
                        depth_samples,
                        width_samples]
        return actions


