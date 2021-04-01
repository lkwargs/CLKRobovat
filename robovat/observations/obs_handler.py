from matplotlib import pyplot as plt


class ObservationHandler(object):
    NUM_FIGURES = 0

    def __init__(self):
        self.num_plots = 0

    def plot(self, _images, _action=None):
        ObservationHandler.NUM_FIGURES += 1
        plt.figure(ObservationHandler.NUM_FIGURES)

        for img in _images:
            self.plot_helper(img, _action)

    def plot_helper(self, _image, grasp=None):
        self.num_plots += 1
        plt.subplot(2, 2, self.num_plots)
        plt.imshow(_image)
        if grasp is not None:
            plt.plot(grasp.center[0], grasp.center[1], "o", color="w")

            axis = grasp.axis
            g1 = grasp.center - float(grasp.width_pixel) * axis
            g2 = grasp.center + float(grasp.width_pixel) * axis

            plt.plot([g1[0], g2[0]], [g1[1], g2[1]], color='r')

    def show(self):
        self.num_plots = 0
        plt.show()
