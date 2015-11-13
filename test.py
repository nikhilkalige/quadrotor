from quadcopter_model import Quadcopter
from plotter import PlotFlight
import numpy as np


def generate_flip_sections(mass, Ixx, length, Bup, Bdown, Cpmax, Cn, gravity):
    """Generate parameters for multiflip action"""
    sections = np.zeros(5, dtype='object')
    p0 = p3 = 0.9 * Bup
    # p1 = p4 = gravity * (T2 + T3 + T4) / (2 * p0)
    p1 = p4 = 0.1

    ap = {
        'acc': (-mass * length * (Bup - p0) / (4 * Ixx)),
        'start': (mass * length * (Bup - Bdown) / (4 * Ixx)),
        'coast': 0,
        'stop': (-mass * length * (Bup - Bdown) / (4 * Ixx)),
        'recover': (mass * length * (Bup - p3) / (4 * Ixx)),
    }

    T2 = (Cpmax - p1 * ap['acc']) / ap['start']
    T4 = -(Cpmax + p4 * ap['recover']) / ap['stop']
    p2 = T3 = (2 * np.pi * Cn / Cpmax) - (Cpmax / ap['start'])

    aq = 0
    ar = 0

    # 1. Accelerate
    sections[0] = (mass * p0, [ap['acc'], aq, ar], p1)

    temp = mass * Bup - 2 * abs(ap['start']) * Ixx / length
    sections[1] = (temp, [ap['start'], aq, ar], T2)

    sections[2] = (mass * Bdown, [ap['coast'], aq, ar], p2)

    temp = mass * Bup - 2 * abs(ap['stop']) * Ixx / length
    sections[3] = (temp, [ap['stop'], aq, ar], T4)

    sections[4] = (mass * p3, [ap['recover'], aq, ar], p4)
    return sections


class MultiFlipParams(object):
    def __init__(self):
        self.mass = 0.468
        self.Ixx = 0.0023
        self.length = 0.17
        self.Bup = 21.58
        self.Bdown = 3.92
        self.Cpmax = np.pi * 1800/180
        self.Cn = 3
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


ideal_final_state = np.array([0, 0, 0, 0, 0, 0, 2 * np.pi * 3, 0, 0])


def cmaes_evaluate(params):
    """5 dimensional variables[p0 ..... p5]"""
    gen = MultiFlipParams()
    quad = Quadcopter()

    sections = gen.get_sections(params)
    quad.update_state(sections)
    final_state = np.array([quad.state['position'],
                            quad.state['velocity'],
                            quad.state['orientation']]).flatten()
    return ideal_final_state - final_state


###################################################
# Test Multi flips
###################################################
def test1():
    quad = Quadcopter()
    # Should Cpmax be 1800deg/s or (np.pi * 1800 / 180) rad/s
    sections = generate_flip_sections(0.468, 0.0023, 0.17, 21.58, 3.92, (np.pi * 1800/180), 3, 9.806)
    state = quad.update_state(sections)
    PlotFlight(state, 0.17).show()


def test2():
    gen = MultiFlipParams()
    quad = Quadcopter()
    params = gen.get_initial_parameters()
    sections = gen.get_sections(params)
    state = quad.update_state(sections)
    PlotFlight(state, 0.17).show()

#test1()
test2()
