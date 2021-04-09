import numpy as np

from robovat.perception.camera.camera import Camera


class SegmentationGraspSampler(object):
    """Segmentation image grasp sampler.

    Use segmentation image to sample grasp actions.
    """

    def __init__(self, gripper_width):
        self.gripper_width = gripper_width


    def sample(self, image, depth, camera, num_samples):

        if not isinstance(camera, Camera):
            intrinsics = camera
            camera = Camera()
            camera.set_calibration(intrinsics, np.zeros((3,)), np.zeros((3,)))

        image = np.squeeze(image, -1)
        depth = np.squeeze(depth, -1)
        objs = np.unique(image)
        obj_exists = np.delete(objs, np.where(objs == 1))
        obj_selected = np.random.randint(0, len(obj_exists))
        print("obj exist: ", obj_exists, "obj select: ", obj_selected)

        # Sample grasp point
        valid_points = np.where(image == obj_exists[obj_selected])
        valid_idx = np.random.choice(valid_points[0].shape[0], num_samples)
        center_samples = np.c_[valid_points[1][valid_idx], valid_points[0][valid_idx]]

        depth_samples = np.array([depth[c[0]][c[1]] for c in center_samples]) + 0.1
        angle_samples = (np.random.random(num_samples) * 2 - 1) * np.pi

        width_samples = np.array([0.05] * num_samples)
        actions = np.c_[center_samples,
                        angle_samples,
                        depth_samples,
                        width_samples]
        return actions
