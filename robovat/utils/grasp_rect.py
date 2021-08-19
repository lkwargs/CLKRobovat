import torch
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max


plate_size = 3
gripper_width = 22


class GraspRectangle(object):

    def __init__(self, points, angle) -> None:
        self.points = np.array(points)
        self.angle = angle
        self.length = 1

    @staticmethod
    def from_grasp2d(center, angle):
        center = np.array([center[1], center[0]])

        # Compute axis and jaw locations.
        axis = np.array([np.cos(angle), np.sin(angle)])
        g1 = center - 0.5 * float(30) * axis
        g2 = center + 0.5 * float(30) * axis

        # Direction of jaw line.
        jaw_dir = plate_size * np.array([axis[1], -axis[0]])

        p1, p2 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T
        p3, p4 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T

        return GraspRectangle([p1, p2, p3, p4], angle)

    @staticmethod
    def load_from_cornell_file(fname, output_shape):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()

                def to_point(p):
                    x, y = p.split()
                    return [int(round(float(x))), int(round(float(y)))]
                try:
                    gr = np.array([
                        to_point(p0),
                        to_point(p1),
                        to_point(p2),
                        to_point(p3)
                    ])
                    gr = np.c_[np.clip(gr[:, 0], 0, output_shape[0]), np.clip(gr[:, 1], 0, output_shape[1])]

                    dx = gr[1, 1] - gr[0, 1]
                    dy = gr[1, 0] - gr[0, 0]
                    angle = (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

                    grs.append(GraspRectangle(gr, angle))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return grs


def rectangles2image(grs, shape, position=True, angle=True, width=True):

    if position:
        pos_out = np.zeros(shape[1:])
    else:
        pos_out = None
    if angle:
        ang_out = np.zeros(shape[1:])
    else:
        ang_out = None
    if width:
        width_out = np.zeros(shape[1:])
    else:
        width_out = None

    for gr in grs:
        x, y = gr.points[:, 0], gr.points[:, 1]
        x, y = [x[0], x[2], x[3], x[1]], [y[0], y[2], y[3], y[1]]

        rr, cc = polygon(x, y, shape[1:])
        if position:
            pos_out[rr, cc] = 1.0
        if angle:
            ang_out[rr, cc] = gr.angle
        if width:
            width_out[rr, cc] = gr.length

        cos = np.cos(ang_out)
        sin = np.sin(ang_out)

    def trans(x):
        return torch.tensor(x.reshape(shape), dtype=torch.float32)

    res = [trans(pos_out), trans(cos), trans(sin), trans(width_out)]

    return res
