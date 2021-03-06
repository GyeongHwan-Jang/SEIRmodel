import os
import sys
sys.path.append("/root/.jupyter/python_file/")

from PyCX import pycxsimulator
from pylab import *

import copy as cp

S_init,I_init = 90, 10
ms, mi = 0.03, 0.03
beta = 0.3

# nr = 500. # carrying capacity of rabbits
#
# r_init = 100 # initial rabbit population
# mr = 0.03 # magnitude of movement of rabbits
# dr = 1.0 # death rate of rabbits when it faces foxes
# rr = 0.1 # reproduction rate of rabbits
#
# f_init = 30 # initial fox population
# mf = 0.05 # magnitude of movement of foxes
# df = 0.1 # death rate of foxes when there is no food
# rf = 0.5 # reproduction rate of foxes

cd = 0.02 # radius for collision detection
cdsq = cd ** 2

class agent:
    pass

def initialize():
    global agents
    agents = []
    for i in range(S_init + I_init):
        ag = agent()
        ag.type = 'S' if i < I_init else 'I'
        ag.x = random()
        ag.y = random()
        agents.append(ag)

def observe():
    global agents
    cla()
    S = [ag for ag in agents if ag.type == 'S']
    if len(S) > 0:
        x = [ag.x for ag in S]
        y = [ag.y for ag in S]
        plot(x, y, 'b.')
    I = [ag for ag in agents if ag.type == 'I']
    if len(I) > 0:
        x = [ag.x for ag in I]
        y = [ag.y for ag in I]
        plot(x, y, 'ro')
    axis('image')
    axis([0, 1, 0, 1])

def update_one_agent():
    global agents
    if agents == []:
        return

    ag = choice(agents)

    # simulating random movement
    m = ms if ag.type == 'S' else mi
    ag.x += uniform(-m, m)
    ag.y += uniform(-m, m)
    ag.x = 1 if ag.x > 1 else 0 if ag.x < 0 else ag.x
    ag.y = 1 if ag.y > 1 else 0 if ag.y < 0 else ag.y

    # detecting collision and simulating death or birth
    neighbors = [nb for nb in agents if nb.type != ag.type
                 and (ag.x - nb.x)**2 + (ag.y - nb.y)**2 < cdsq]

    if ag.type == 'S':
        if len(neighbors) > 0: # if there are foxes nearby
            if random() < beta:
                ag.type = 'I'
                return
    # else:
    #     if len(neighbors) == 0: # if there are no rabbits nearby
    #         if random() < df:
    #             agents.remove(ag)
    #             return
    #     else: # if there are rabbits nearby
    #         if random() < rf:
    #             agents.append(cp.copy(ag))

def update():
    global agents
    t = 0.
    while t < 1. and len(agents) > 0:
        t += 1. / len(agents)
        update_one_agent()

pycxsimulator.GUI().start(func=[initialize, observe, update])
