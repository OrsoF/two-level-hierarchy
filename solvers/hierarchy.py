import numpy as np
import mdptoolbox

from solvers.utils import get_periphery, get_boundary, normalize, find_index, general_to_show


class Regions:
    def __init__(self, env, kappa, method='policy_iteration'):
        self.P = env.P
        self.R = env.R
        self.S = env.S
        self.A = env.A
        self.kappa = kappa
        self.regions = env.regions
        self.method = method

        self.global_S = len(self.regions)
        self.global_A = len(self.regions) ** 2

        self.global_policy = [0 for _ in range(self.global_S)]
        self.peripheries = [get_periphery(self.P, region) for region in self.regions]
        self.boundaries = [get_boundary(self.P, region) for region in self.regions]
        self.completed_regions = [self.regions[i] + self.peripheries[i] for i in range(self.global_S)]

        self.global_actions = []
        self.local_transitions = []
        self.local_rewards = []
        self.local_policies = []
        for r1 in range(self.global_S):
            for r2 in range(self.global_S):
                self.global_actions.append((r1, r2))
                p, r = self.build_local_p(r1), self.build_local_r(r1, r2)
                self.local_transitions.append(p)
                self.local_rewards.append(r)
                if self.method == 'policy_iteration':
                    vi = mdptoolbox.mdp.PolicyIteration(p, r, 0.99999, max_iter=10000)
                else:
                    vi = mdptoolbox.mdp.ValueIteration(p, r, 0.99999, max_iter=10000)
                vi.run()
                policy = np.array(vi.policy).astype(int)
                self.local_policies.append(policy)

        self.global_P = np.zeros((self.global_A, self.global_S, self.global_S))
        self.global_R = np.zeros((self.global_A, self.global_S, self.global_S))

        self.build_global_mdp()
        self.solve_global_mdp()

    def build_local_r(self, r1, r2):
        A = self.A
        region = self.regions[r1]
        state_space = self.completed_regions[r1]
        region_final = self.regions[r2]
        S = len(state_space)
        r = np.zeros((A, S, S))

        for action in range(A):
            for ini_state_index, ini_state in enumerate(state_space):
                for fin_state_index, fin_state in enumerate(state_space):
                    if self.P[action, ini_state, fin_state] > 0:
                        if ini_state in region:
                            if fin_state in region:
                                r[action, ini_state_index, fin_state_index] = self.R[action, ini_state, fin_state]
                            else:
                                if fin_state in region_final:
                                    r[action, ini_state_index, fin_state_index] = self.R[action, ini_state, fin_state]
                                else:
                                    r[action, ini_state_index, fin_state_index] = self.kappa \
                                                                                  + self.R[action, ini_state, fin_state]
        return r

    def build_local_p(self, r1):
        A = self.A
        region = self.regions[r1]
        periph = self.peripheries[r1]
        state_space = self.completed_regions[r1]
        S = len(state_space)
        p = np.zeros((A, S, S))

        for action in range(A):
            for ini_state_index, ini_state in enumerate(state_space):
                for fin_state_index, fin_state in enumerate(state_space):
                    if ini_state in region:
                        p[action, ini_state_index, fin_state_index] = self.P[action, ini_state, fin_state]
                    elif ini_state in periph:
                        p[action, ini_state_index, ini_state_index] = 1
        return p

    def region_transition_possible(self, r1, r2):
        for ini_state in self.regions[r1]:
            for fin_state in self.regions[r2]:
                if np.max(self.P[:, ini_state, fin_state]):
                    return True
        return False

    def global_cost_function(self, action, starting_region, reached_region):
        action_origin, action_destination = self.global_actions[action]

        if not starting_region == action_origin or not self.region_transition_possible(action_origin,
                                                                                       action_destination):
            return -10000

        R, S = starting_region, reached_region
        R1, R2 = action_origin, action_destination

        size_region = len(self.regions[R1])
        policy = self.local_policies[action]

        P = np.zeros((size_region, size_region))
        for state_index_1, state_1 in enumerate(self.regions[R1]):
            for state_index_2, state_2 in enumerate(self.regions[R1]):
                P[state_index_1, state_index_2] = self.P[policy[state_index_1], state_1, state_2]

        A = np.eye(size_region) - P

        b = np.zeros(size_region)
        for state_index_1, state_1 in enumerate(self.regions[R]):
            b[state_index_1] += sum(int(state_2 in self.regions[S])
                                    * int(state_2 in self.peripheries[R])
                                    * self.R[policy[state_index_1], state_1, state_2]
                                    * self.P[policy[state_index_1], state_1, state_2]
                                    for state_2 in range(self.S))
            b[state_index_1] += sum(self.R[policy[state_index_1], state_1, state_2]
                                    * self.P[policy[state_index_1], state_1, state_2]
                                    for state_index_2, state_2 in enumerate(self.regions[R]))

        #         A = np.concatenate([A, np.ones((1, size_region))])
        #         b = np.concatenate([b, np.ones(1)])

        phi = np.linalg.lstsq(A, b, rcond=None)[0]
        return sum(int(state in self.boundaries[R]) * phi[state_index] for state_index, state in
                   enumerate(self.regions[R])) / len(self.boundaries[R])

    def global_transition_function(self, action, starting_region, reached_region):
        action_origin, action_destination = self.global_actions[action]

        if not starting_region == action_origin or not self.region_transition_possible(action_origin,
                                                                                       action_destination):
            if starting_region == reached_region:
                return 1
            else:
                return 0

        R, S = starting_region, reached_region
        r1, r2 = action_origin, action_destination

        size_region = len(self.regions[r1])
        policy = self.local_policies[action]

        P = np.zeros((size_region, size_region))
        for state_index_1, state_1 in enumerate(self.regions[r1]):
            for state_index_2, state_2 in enumerate(self.regions[r1]):
                P[state_index_1, state_index_2] = self.P[policy[state_index_1], state_1, state_2]

        A = np.eye(size_region) - P

        b = np.zeros(size_region)
        for state_index_1, state_1 in enumerate(self.regions[r1]):
            b[state_index_1] = sum(int(state_2 in self.regions[S])
                                   * int(state_2 in self.peripheries[R])
                                   * self.P[policy[state_index_1], state_1, state_2]
                                   for state_2 in range(self.S))

        A = np.concatenate([A, np.ones((1, size_region))])
        b = np.concatenate([b, np.ones(1)])
        phi = np.linalg.lstsq(A, b, rcond=None)[0]
        return sum(int(state in self.boundaries[R])
                   * phi[state_index]
                   for state_index, state in enumerate(self.regions[R])) / len(self.boundaries[R])

    def build_global_mdp(self):
        for action in range(self.global_A):
            for R in range(self.global_S):
                for S in range(self.global_S):
                    self.global_P[action, R, S] = self.global_transition_function(action, R, S)
                    self.global_R[action, R, S] = self.global_cost_function(action, R, S)

    def solve_global_mdp(self):
        if self.method == 'policy_iteration':
            vi = mdptoolbox.mdp.PolicyIteration(normalize(self.global_P), self.global_R, 0.99999, max_iter=10000)
        else:
            vi = mdptoolbox.mdp.ValueIteration(normalize(self.global_P), self.global_R, 0.99999, max_iter=10000)
        vi.run()
        self.global_policy = vi.policy

    def find_region(self, state):
        for region_index, region in enumerate(self.regions):
            if state in region:
                return region_index

    def evolve(self, state):
        current_region = self.find_region(state)
        global_action = self.global_policy[current_region]

        local_policy = self.local_policies[global_action]

        p, r = self.local_transitions[global_action], self.local_rewards[global_action]

        index_in_region = find_index(state, self.regions[current_region])
        action = local_policy[index_in_region]
        next_state = np.random.choice(a=self.completed_regions[current_region], p=p[action, index_in_region])

        return next_state, self.R[action, state, next_state]

    def test_solution(self):
        results = np.zeros((10, 10))
        for state in range(100):
            i, j = general_to_show(state)
            total_reward = 0
            for k in range(1000):
                state, reward = self.evolve(state)
                total_reward += reward
            if state == 2:
                results[i, j] = 1
        print(results)
        print(np.sum(results), ' %')
