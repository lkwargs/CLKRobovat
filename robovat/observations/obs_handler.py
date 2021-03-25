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

    def plot_helper(self, _image, _action=None):
        self.num_plots += 1
        plt.subplot(2, 2, self.num_plots)
        plt.imshow(_image)
        if _action is not None:
            plt.plot(_action[0], _action[1], ".", color="r")

    def show(self):
        self.num_plots = 0
        plt.show()
