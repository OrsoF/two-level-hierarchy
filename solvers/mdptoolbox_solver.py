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


def evolve_iteration_solver(p, r, policy, state):
    action = policy[state]
    next_state = np.random.choice(a=range(p.shape[1]), p=p[action, state])
    reward = r[action, state, next_state]
    return next_state, action, reward


def trajectory(p, r, policy, state, size=1000):
    states, actions, rewards = [state], [], []
    for i in range(size):
        state, action, reward = evolve_iteration_solver(p, r, policy, state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
    return states, actions, rewards


def test_policy(p, r, policy):
    success = 0
    success_states = np.zeros((p.shape[1]))
    for state in range(p.shape[1]):
        success += int(trajectory(p, r, policy, state)[0][-1] == 2)
        if int(trajectory(p, r, policy, state)[0][-1] == 2):
            success_states[state] = 1
    return success/p.shape[1], success_states.reshape((10, 10))
