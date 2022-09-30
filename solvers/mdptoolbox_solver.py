import mdptoolbox
import numpy as np


def solve(p, r, gamma=0.99, methods='value_iteration', verbose=False):
    if methods == 'value_iteration':
        vi = mdptoolbox.mdp.ValueIteration(p, r, gamma)
    else:
        vi = mdptoolbox.mdp.PolicyIteration(p, r, gamma)
    if verbose:
        vi.setVerbose()
    vi.run()
    return vi


def action_to_cardinal_direction(a):
    if a == 0:
        return 'N'
    elif a == 1:
        return 'S'
    elif a == 2:
        return 'E'
    else:
        return 'W'


def show_policy_matrix(policy):
    policy = [action_to_cardinal_direction(a) for a in policy]
    mat = np.zeros((10, 10), dtype=str)
    for i in range(10):
        for j in range(10):
            mat[i, j] = policy[10 * i + j]

    print(mat)
