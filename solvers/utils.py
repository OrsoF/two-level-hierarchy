import numpy as np
import sys


def normalize(p):
    p = np.round(p, 4)
    for action in range(p.shape[0]):
        for state in range(p.shape[1]):
            if np.sum(p[action, state]) == 0:
                print('Problem in building the transition matrix : zero-zum row.')
                sys.exit()
            else:
                p[action, state, state] += 1 - np.sum(p[action, state])
    return p


def find_index(state, region):
    for state_index_1, state_1 in enumerate(region):
        if state_1 == state:
            return state_index_1
    raise Exception


def general_to_show(s):
    return s // 10, s % 10


def analyze_P_matrix(p):
    for action in range(p.shape[0]):
        for state in range(p.shape[1]):
            if np.sum(p[action, state]) < 1-10**-5:
                print('Problem : ')
                print('Sum : ', np.sum(p[action, state]))
                print('Action : ', action)
                print('State : ', state)
                return
    print('P is stochastic')
