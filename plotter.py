import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from mpl_toolkits import mplot3d


class PlotFlight(object):
    def __init__(self, state, arm_length):
        self.state = state
        self.length = len(state)
        self.arm_length = arm_length

    def setup_plot(self):
        # setup
        self.fig = pyplot.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

        # plot settings
        pyplot.axis("equal")

        # plot labels
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # plot limits
        plt_limits = np.array([[self.state[:, 0].min(), self.state[:, 0].max()],
                               [self.state[:, 1].min(), self.state[:, 1].max()],
                               [self.state[:, 2].min(), self.state[:, 2].max()]])
        for item in plt_limits:
            if abs(item[1] - item[0]) < 2:
                item[0] -= 1
                item[1] += 1

        self.ax.set_xlim3d(plt_limits[0])
        self.ax.set_ylim3d(plt_limits[1])
        self.ax.set_zlim3d(plt_limits[2])
        # self.ax.set_xlim3d([-10, 10])
        # self.ax.set_ylim3d([-10, 10])
        # self.ax.set_zlim3d([-10, 10])

        # initialize the plot
        flight_path, = self.ax.plot([], [], [], '--')
        arms = [self.ax.plot([], [], [], c='r', marker='^')[0] for _ in range(4)]
        self.plot_artists = [flight_path, arms]

    def rotate(self, euler_angles, point):
        [phi, theta, psi] = euler_angles
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        m = np.array([[cthe * cpsi, sphi * sthe * cpsi - cphi * spsi, cphi * sthe * cpsi + sphi * spsi],
                      [cthe * spsi, sphi * sthe * spsi + cphi * cpsi, cphi * sthe * spsi - sphi * cpsi],
                      [-sthe,       cthe * sphi,                      cthe * cphi]])

        return np.dot(m, point)

    def init_animate(self):
        self.plot_artists[0].set_data([], [])
        self.plot_artists[0].set_3d_properties([])

        for arm in self.plot_artists[1]:
            arm.set_data([], [])
            arm.set_3d_properties([])

        return [self.plot_artists[0]] + self.plot_artists[1]

    def animate(self, i):
        i = (i + 1) % (self.length + 1)
        x = self.state[:, 0][:i]
        y = self.state[:, 1][:i]
        z = self.state[:, 2][:i]

        center_point = np.array([x[-1], y[-1], z[-1]])
        euler_angles = self.state[i - 1][6:9]

        self.plot_artists[0].set_data(x, y)
        self.plot_artists[0].set_3d_properties(z)

        arm_base_pos = np.array([[self.arm_length, 0, 0],
                                 [0, -self.arm_length, 0],
                                 [-self.arm_length, 0, 0],
                                 [0, self.arm_length, 0]])

        arm_base_pos = [self.rotate(euler_angles, arm) for arm in arm_base_pos]

        # update the position
        arm_base_pos = [(arm + center_point) for arm in arm_base_pos]
        self.plot_arms(center_point, arm_base_pos)
        return [self.plot_artists[0]] + self.plot_artists[1]

    def plot_arms(self, center, arm_pos):
        arm_lines = self.plot_artists[1]
        for index, arm in enumerate(arm_pos):
            pos = np.column_stack((center, arm))
            arm_lines[index].set_data(pos[:2])
            arm_lines[index].set_3d_properties(pos[-1:])

    def show(self):
        self.setup_plot()
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init_animate,
                                       frames=self.length, interval=1,
                                       blit=True)
        pyplot.gca().set_aspect("equal", adjustable="box")
        pyplot.show()
