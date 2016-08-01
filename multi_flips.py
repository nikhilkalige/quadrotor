import multiprocessing
from quadcopter_model import Quadcopter
from plotter import PlotFlight
import numpy as np
import random
import time
from deap import base, creator, tools, algorithms, cma
import argparse
from matplotlib import pyplot


TURNS = 3


class PlotCMAES(object):
    graph_lengths = [9, 5, 9, 9]
    titles = [
        'Std Deviation - fitness (NGEN)',
        'Parameters (NGEN)',
        'Fitness (Current Gen)',
        'Fitness (NGEN)'
    ]
    xlabel = ['No. Generations', 'No. Generations', 'No. Children', 'No. Generations']
    labels = [
        ['x', 'y', 'z', 'xdot', 'ydot', 'zdot', 'phi', 'theta', 'psi'],
        ['Facc', 'Tacc', 'Tcoa', 'Frec', 'Trec']
    ]
    markers = [
        ['-', '-', '-', '--', '--', '--', '-.', '-.', '-.'],
        ['-', '-.', '-', '-.', '-']
    ]

    def __init__(self, ngen, children):
        self.fig, self.axis_arr = pyplot.subplots(2, 2)
        self.axis_arr = self.axis_arr.flatten()
        self.lines = [[] for _ in xrange(4)]

        for i, length in enumerate(self.graph_lengths):
            label = self.labels[1] if i == 1 else self.labels[0]
            marker = self.markers[1] if i == 1 else self.markers[0]
            for j in xrange(length):
                self.lines[i].append(self.axis_arr[i].plot([], [],
                                                           label=label[j],
                                                           linewidth=3,
                                                           linestyle=marker[j])[0])

        for i, axis in enumerate(self.axis_arr):
            axis.grid(True)
            axis.set_title(self.titles[i])
            axis.set_xlabel(self.xlabel[i])
            axis.legend(loc='upper right')

        self.axis_arr[0].set_yscale('symlog')
        self.axis_arr[2].set_yscale('symlog')
        self.axis_arr[3].set_yscale('symlog')

        for i in [0, 1, 3]:
            self.axis_arr[i].set_autoscaley_on(True)
            self.axis_arr[i].set_xlim(0, ngen)

        self.axis_arr[2].set_autoscaley_on(True)
        self.axis_arr[2].set_xlim(0, children)

        pyplot.grid(True)
        pyplot.ion()
        pyplot.show()

    def update(self, plot1, plot2, plot3, plot4):
        data = [plot1, plot2, plot3, plot4]
        xlen = [plot1.shape[0], len(plot2), plot3.shape[0], plot1.shape[0]]
        for i in xrange(4):
            # print 'I', i, xlen
            # print data[i]
            for line, y in zip(self.lines[i], data[i].T):
                line.set_ydata(y)
                line.set_xdata(range(xlen[i]))

        for i in xrange(4):
            self.axis_arr[i].relim()
            self.axis_arr[i].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class MultiFlipParams(object):
    """Generates the parameters for CMA-ES according to the paper."""
    def __init__(self):
        self.mass = 0.468
        self.Ixx = 0.0023
        self.length = 0.17
        self.Bup = 21.58
        self.Bdown = 3.92
        self.Cpmax = np.pi * 1800/180
        self.Cn = TURNS
        self.gravity = 9.806

    def get_acceleration(self, p0, p3):
        """Compute the acceleration from the generated parameters."""
        ap = {
            'acc': (-self.mass * self.length * (self.Bup - p0) / (4 * self.Ixx)),
            'start': (self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'coast': 0,
            'stop': (-self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'recover': (self.mass * self.length * (self.Bup - p3) / (4 * self.Ixx)),
        }
        return ap

    def get_initial_parameters(self):
        """Initial parameters."""
        p0 = p3 = 0.9 * self.Bup
        p1 = p4 = 0.1
        acc_start = self.get_acceleration(p0, p3)['start']
        p2 = (2 * np.pi * self.Cn / self.Cpmax) - (self.Cpmax / acc_start)
        return [p0, p1, p2, p3, p4]

    def get_sections(self, parameters):
        """Compute the 5 regions of the flight as defined in the paper."""
        sections = np.zeros(5, dtype='object')
        [p0, p1, p2, p3, p4] = parameters

        ap = self.get_acceleration(p0, p3)

        T2 = (self.Cpmax - p1 * ap['acc']) / ap['start']
        T4 = -(self.Cpmax + p4 * ap['recover']) / ap['stop']

        aq = 0
        ar = 0

        # 1. Accelerate
        sections[0] = (self.mass * p0, [ap['acc'], aq, ar], p1)

        temp = self.mass * self.Bup - 2 * abs(ap['start']) * self.Ixx / self.length
        sections[1] = (temp, [ap['start'], aq, ar], T2)

        sections[2] = (self.mass * self.Bdown, [ap['coast'], aq, ar], p2)

        temp = self.mass * self.Bup - 2 * abs(ap['stop']) * self.Ixx / self.length
        sections[3] = (temp, [ap['stop'], aq, ar], T4)

        sections[4] = (self.mass * p3, [ap['recover'], aq, ar], p4)
        return sections


ideal_final_state = np.array([0, 0, 0, 0, 0, 0, 2 * np.pi * TURNS, 0, 0])


def cmaes_evaluate(params):
    """5 dimensional variables[p0 ..... p5]"""
    gen = MultiFlipParams()
    quad = Quadcopter(False)
    # print("cmaes_evaluate")
    # Restrict duration of generated params to below 2 seconds
    if (params[1] > 2.5) or (params[2] > 2.5) or (params[4] > 2.5):
        return tuple([1000000] * 9)
    sections = gen.get_sections(params)
    for sect in sections:
        if sect[2] < 0:
            # print('Error sect:', sect)
            return tuple([1000000] * 9)

    quad.update_state(sections)
    final_state = np.array([quad.state['position'],
                            quad.state['velocity'],
                            quad.state['orientation']]).flatten()
    fitness = abs(ideal_final_state - final_state)
    # print("[", params, "] -> [", fitness, "]")
    return tuple(fitness)


def fly_quadrotor(params=None, fly=True):
    gen = MultiFlipParams()
    quad = Quadcopter()
    if not params:
        params = gen.get_initial_parameters()
    sections = gen.get_sections(params)
    state = quad.update_state(sections)
    if fly:
        PlotFlight(state, 0.17).show()
    return state


def run_cmaes():
    # search_space_dims = 5
    NGEN = 1000
    CHILD = 6
    SIGMA = 1
    verbose = True

    gen = MultiFlipParams()
    cmplot = PlotCMAES(NGEN, CHILD)
    random.seed()

    best_params = np.ndarray((NGEN, 5))
    best_fitness = np.ndarray((NGEN, 9))

    print('Init params:', gen.get_initial_parameters())

    # The fitness function should minimize all the 9 variables
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0,
                                                        -1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    pool = multiprocessing.Pool(3)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", cmaes_evaluate)
    toolbox.register("map", pool.map)

    cma_es = cma.Strategy(centroid=gen.get_initial_parameters(), sigma=SIGMA, lambda_=CHILD)
    toolbox.register("generate", cma_es.generate, creator.Individual)
    toolbox.register("update", cma_es.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    start = time.time()

    # Since we are doing addtional work like plotting, implement the
    # algorithm.eaGenerateUpdate part yourself
    logbook = tools.Logbook()
    # logbook.header = ['gen'] + stats.fields
    logbook.header = ['gen']

    for gen in range(NGEN):
        population = toolbox.generate()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        toolbox.update(population)
        hof.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)
        if verbose:
            print(logbook.stream)

        plot1 = np.asarray(logbook.select("std"))
        # Holds the best parameter set for each generation
        best_params[gen] = hof[0]
        # Fitness of current population
        plot3 = np.asarray([ind.fitness.values for ind in population])
        # Fitness of best population over all generations
        best_fitness[gen] = hof[0].fitness.values
        cmplot.update(plot1, best_params[:gen+1], plot3, best_fitness[:gen+1])

    print("Best individual is %s, fitness: %s" % (hof[0], hof[0].fitness.values))
    print("Elapsed %s minutes" % ((time.time() - start)/60.0))

    pyplot.show(True)
    # Fly the quadrotor with generated params
    fly_quadrotor(hof[0])
    pyplot.show(True)


def load_data(f):
    data = f.readline().strip().strip("[]").split(",")
    return [float(i) for i in data]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quadcopter multiflips")
    parser.add_argument("-f", nargs="?", type=argparse.FileType('r'),
                        help="Parameters file to generate output")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fly", action='store_true',
                       help="Plot the flight")
    group.add_argument("--cmaes", action='store_true',
                       help="Run cmaes optimization")
    group.add_argument("--blender", action='store_true',
                       help="Generate data for blender")
    args = parser.parse_args()

    params = None
    if args.f:
        params = load_data(args.f)

    if params and len(params) != 5:
        params = None

    if args.fly:
        fly_quadrotor(params)

    if args.cmaes:
        run_cmaes()

    if args.blender:
        state = fly_quadrotor(params, fly=False)
        np.save("quadata", state)
        print("Output save to quadata.npy")
