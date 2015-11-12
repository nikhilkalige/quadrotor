from quadcopter_model import Quadcopter
import numpy as np
from matplotlib import pyplot
from matplotlib import animation
from mpl_toolkits import mplot3d
from plotter import PlotFlight


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


def generate_translation_motion():
    """Generate parameters for translation motion in yz plane.
    From (y1,z1) to (y2, z2)
    """
    pass




###################################################
# Test Multi flips
###################################################
quad = Quadcopter()
# Should Cpmax be 1800deg/s or (np.pi * 1800 / 180) rad/s
sections = generate_flip_sections(0.468, 0.0023, 0.17, 21.58, 3.92, (np.pi * 1800/180), 3, 9.806)
state = quad.update_state(sections)
PlotFlight(state)
