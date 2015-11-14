import multiprocessing
from quadcopter_model import Quadcopter
from plotter import PlotFlight
import numpy as np
import random
import time
from deap import base, creator, tools, algorithms, cma


TURNS = 3


class MultiFlipParams(object):
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
        ap = {
            'acc': (-self.mass * self.length * (self.Bup - p0) / (4 * self.Ixx)),
            'start': (self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'coast': 0,
            'stop': (-self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'recover': (self.mass * self.length * (self.Bup - p3) / (4 * self.Ixx)),
        }
        return ap

    def get_initial_parameters(self):
        p0 = p3 = 0.9 * self.Bup
        p1 = p4 = 0.1
        acc_start = self.get_acceleration(p0, p3)['start']
        p2 = (2 * np.pi * self.Cn / self.Cpmax) - (self.Cpmax / acc_start)
        return [p0, p1, p2, p3, p4]

    def get_sections(self, parameters):
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
    quad = Quadcopter()
    # print "cmaes_evaluate"
    sections = gen.get_sections(params)
    for sect in sections:
        if sect[2] < 0:
            # print 'Error sect:', sect
            return tuple([1000000] * 9)

    quad.update_state(sections)
    final_state = np.array([quad.state['position'],
                            quad.state['velocity'],
                            quad.state['orientation']]).flatten()
    fitness = abs(ideal_final_state - final_state)
    # print "[", params, "] -> [", fitness, "]"
    return tuple(fitness)


def fly_quadrotor(params=None):
    gen = MultiFlipParams()
    quad = Quadcopter()
    if not params:
        params = gen.get_initial_parameters()
    sections = gen.get_sections(params)
    state = quad.update_state(sections)
    PlotFlight(state, 0.17).show()


def run_cmaes():
    # search_space_dims = 5
    gen = MultiFlipParams()
    random.seed()

    print 'Init params:', gen.get_initial_parameters()
    # The fitness function should minimize all the 9 variables
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.1, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    pool = multiprocessing.Pool(2)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", cmaes_evaluate)
    toolbox.register("map", pool.map)

    cma_es = cma.Strategy(centroid=gen.get_initial_parameters(), sigma=3)
    toolbox.register("generate", cma_es.generate, creator.Individual)
    toolbox.register("update", cma_es.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    start = time.time()
    # The CMA-ES algorithm converge with good probability with those settings
    pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=60, stats=stats,
                                               halloffame=hof, verbose=False)

    print("Best individual is %s, fitness: %s" % (hof[0], hof[0].fitness.values))
    print("Elapsed %s minutes" % ((time.time() - start)/60.0))

    # Fly the quadrotor with generated params
    fly_quadrotor(hof[0])



#########
# test functions
########

fly_quadrotor()
#run_cmaes()
