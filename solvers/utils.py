import numpy as np
import sys


def normalize(p):
    for action in range(p.shape[0]):
        for state in range(p.shape[1]):
            if np.sum(p[action, state]) == 0:
                print('Problem in building the transition matrix : zero-zum row.')
                sys.exit()
            else:
                p[action, state] = p[action, state] / np.sum(p[action, state])
    return p


def region_leads_to_state(p, state, region):
    for a in range(p.shape[0]):
        for region_state in region:
            if p[a, region_state, state]:
                return True
    return False


def get_periphery(p, region):
    periphery = []
    for state in range(p.shape[1]):
        if region_leads_to_state(p, state, region) and state not in region:
            periphery.append(state)
    return sorted(periphery)


def get_boundary(p, region):
    boundary = []
    periphery = get_periphery(p, region)
    for region_state in region:
        for action in range(p.shape[0]):
            for state in periphery:
                if p[action, region_state, state]:
                    boundary.append(region_state)
    return sorted(boundary)


def find_index(state, region):
    for state_index_1, state_1 in enumerate(region):
        if state_1 == state:
            return state_index_1
    raise Exception


def general_to_show(s):
    return s // 10, s % 10
