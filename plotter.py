import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from mpl_toolkits import mplot3d


class PlotFlight(object):
    def __init__(self, state):
        self.state = state
        self.length = len(state)

        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        self.line, = ax.plot([], [], [], '-')
        self.pt, = ax.plot([], [], [], 'o')

        ax.set_xlim3d([self.state[:, 0].min(), self.state[:, 0].max()])
        ax.set_xlabel('X')

        ax.set_ylim3d([self.state[:, 1].min(), self.state[:, 1].max()])
        ax.set_ylabel('Y')

        ax.set_zlim3d([self.state[:, 2].min(), self.state[:, 2].max()])
        ax.set_zlabel('Z')
        anim = animation.FuncAnimation(fig, self.animate,
                                       init_func=self.init_animate,
                                       frames=self.length, interval=1,
                                       blit=True)
        pyplot.show()

    def init_animate(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])

        self.pt.set_data([], [])
        self.pt.set_3d_properties([])
        return [self.line, self.pt]

    def animate(self, i):
        i = (i + 1) % (self.length + 1)
        x = self.state[:, 0][:i]
        y = self.state[:, 1][:i]
        z = self.state[:, 2][:i]

        self.line.set_data(x, y)
        self.line.set_3d_properties(z)

        self.pt.set_data(x[-1:], y[-1:])
        self.pt.set_3d_properties(z[-1:])
        return [self.line, self.pt]
