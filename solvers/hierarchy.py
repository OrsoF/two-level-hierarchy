import numpy as np
import mdptoolbox


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


def local_transition(params, abstract_action, action, start_state, final_state):
    regions = params['regions']
    transition_matrix = params['transition_matrix']
    start_region = regions[abstract_action[0]]
    if start_state in start_region:
        return transition_matrix[action, start_state, final_state]
    else:
        if start_state == final_state:
            return 1
        else:
            return 0


def local_transition_matrix(params, abstract_action):
    A = params['action_dim']
    start_region_index = abstract_action[0]
    completed_region = params['completed_regions'][start_region_index]
    S = len(completed_region)
    local_matrix = np.zeros((A, S, S))
    for a in range(A):
        for s1 in range(S):
            for s2 in range(S):
                state1 = completed_region[s1]
                state2 = completed_region[s2]
                local_matrix[a, s1, s2] = local_transition(params,
                                                           abstract_action,
                                                           a,
                                                           state1,
                                                           state2)
    return local_matrix


def local_reward(params, abstract_action, action, start_state, final_state):
    start_region_index, final_region_index = abstract_action
    regions = params['regions']
    reward_matrix = params['reward_matrix']
    kappa = params['kappa']
    start_region, final_region = regions[start_region_index], regions[final_region_index]
    if start_state in start_region:
        if final_state in start_region:
            return reward_matrix[action, start_state, final_state]
        elif final_state in final_region:
            return reward_matrix[action, start_state, final_state]
        else:
            return reward_matrix[action, start_state, final_state] + kappa
    else:
        return 0


def local_reward_matrix(params, abstract_action):
    start_region_index, origin_region_index = abstract_action
    transition_matrix = params['transition_matrix']
    completed_regions = params['completed_regions']
    A = transition_matrix.shape[0]
    completed_region = completed_regions[start_region_index]
    S = len(completed_region)
    reward_matrix = np.zeros((A, S, S))
    for a in range(A):
        for s1 in range(S):
            for s2 in range(S):
                state1 = completed_region[s1]
                state2 = completed_region[s2]
                reward_matrix[a, s1, s2] = local_reward(params,
                                                        abstract_action,
                                                        a,
                                                        state1,
                                                        state2)
    return reward_matrix


def get_policy(transition_matrix: np.array,
               reward_matrix: np.array):
    assert len(transition_matrix.shape) == 3 and len(reward_matrix.shape) == 3
    vi = mdptoolbox.mdp.ValueIteration(transition_matrix,
                                       reward_matrix,
                                       0.9999)
    vi.run()
    return np.array(vi.policy).astype(int)


def possible_abstract_transition(params, abstract_state_1, abstract_state_2):
    transition_matrix = params['transition_matrix']
    region_1 = params['regions'][abstract_state_1]
    region_2 = params['regions'][abstract_state_2]
    for ini_state in region_1:
        for fin_state in region_2:
            if np.max(transition_matrix[:, ini_state, fin_state]):
                return True
    return False


class Hierarchy:
    def __init__(self, env, kappa=-4, gamma=0.99):
        self.params = {'transition_matrix': env.P,
                       'reward_matrix': env.R,
                       'state_dim': env.S,
                       'action_dim': env.A,
                       'kappa': kappa,
                       'regions': env.regions,
                       'gamma': gamma}

        self.params['abstract_state_dim'] = len(self.params['regions'])
        self.params['abstract_action_dim'] = len(self.params['regions']) ** 2
        self.params['peripheries'] = [get_periphery(self.params['transition_matrix'], region)
                                      for region in self.params['regions']]
        self.params['boundaries'] = [get_boundary(self.params['transition_matrix'], region)
                                     for region in self.params['regions']]
        self.params['completed_regions'] = [self.params['regions'][i] + self.params['peripheries'][i]
                                            for i in range(self.params['abstract_state_dim'])]
        self.params['abstract_action_set'] = [(r1, r2)
                                              for r1 in range(self.params['abstract_state_dim'])
                                              for r2 in range(self.params['abstract_state_dim'])]

        self.params['local_transition_matrices'] = [local_transition_matrix(self.params, abstract_action)
                                                    for abstract_action in self.params['abstract_action_set']]

        self.params['local_reward_matrices'] = [local_reward_matrix(self.params, abstract_action)
                                                for abstract_action in self.params['abstract_action_set']]

        self.params['local_policies'] = [get_policy(self.params['local_transition_matrices'][i],
                                                    self.params['local_reward_matrices'][i])
                                         for i in range(self.params['abstract_action_dim'])]

        self.params['abstract_transition_matrix'] = np.zeros((self.params['abstract_action_dim'],
                                                              self.params['abstract_state_dim'],
                                                              self.params['abstract_state_dim']))
        for abstract_action_index in range(self.params['abstract_action_dim']):
            for start_abstract_state_index in range(self.params['abstract_state_dim']):
                for final_abstract_state_index in range(self.params['abstract_state_dim']):
                    abstract_action = self.params['abstract_action_set'][abstract_action_index]
                    if abstract_action[0] != start_abstract_state_index:
                        continue
                    policy = self.params['local_policies'][abstract_action_index]

                    size = len(self.params['regions'][start_abstract_state_index])

                    b = []
                    for i_R in self.params['regions'][start_abstract_state_index]:
                        b_i = 0
                        i_R_index = self.params['completed_regions'][start_abstract_state_index].index(i_R)
                        for j in range(self.params['state_dim']):
                            if j in self.params['regions'][final_abstract_state_index] \
                                    and j in self.params['peripheries'][start_abstract_state_index]:
                                b_i += self.params['transition_matrix'][policy[i_R_index], i_R, j]
                        b.append(b_i)
                    b = np.array(b).astype(float)

                    A = np.zeros((size, size))
                    for i in range(size):
                        for j in range(size):
                            action = policy[i]
                            i_global_index = self.params['regions'][start_abstract_state_index][i]
                            j_global_index = self.params['regions'][start_abstract_state_index][j]
                            A[i, j] = self.params['transition_matrix'][action, i_global_index, j_global_index]

                    A = np.eye(size) - A

                    A = np.concatenate([A, np.ones((1, size))]).astype(float)
                    b = np.concatenate([b, np.ones(1)]).astype(float)

                    phi = np.linalg.inv(A.T @ A) @ A.T @ b

                    proba = 0

                    for index, state_region in enumerate(self.params['regions'][start_abstract_state_index]):
                        if state_region in self.params['boundaries'][start_abstract_state_index]:
                            proba += phi[index]

                    self.params['abstract_transition_matrix'][abstract_action_index,
                                                              start_abstract_state_index,
                                                              final_abstract_state_index] = proba
                if np.sum(self.params['abstract_transition_matrix'][abstract_action_index,
                                                                    start_abstract_state_index]) == 0:
                    self.params['abstract_transition_matrix'][abstract_action_index,
                                                              start_abstract_state_index,
                                                              start_abstract_state_index] = 1
                else:
                    self.params['abstract_transition_matrix'][abstract_action_index,
                                                              start_abstract_state_index] \
                        /= np.sum(self.params['abstract_transition_matrix'][abstract_action_index,
                                                                            start_abstract_state_index])

        self.params['abstract_reward_matrix'] = -10000 * np.ones((self.params['abstract_action_dim'],
                                                                  self.params['abstract_state_dim'],
                                                                  self.params['abstract_state_dim']))

        for abstract_action_index in range(self.params['abstract_action_dim']):
            for start_abstract_state_index in range(self.params['abstract_state_dim']):
                for final_abstract_state_index in range(self.params['abstract_state_dim']):
                    abstract_action = self.params['abstract_action_set'][abstract_action_index]
                    if abstract_action[0] != start_abstract_state_index:
                        continue
                    if not possible_abstract_transition(self.params, abstract_action[0], abstract_action[1]):
                        continue
                    policy = self.params['local_policies'][abstract_action_index]

                    size = len(self.params['regions'][start_abstract_state_index])

                    b = []
                    for i_R in self.params['regions'][start_abstract_state_index]:
                        b_i = 0
                        i_R_index = self.params['completed_regions'][start_abstract_state_index].index(i_R)
                        for j in range(self.params['state_dim']):
                            if j in self.params['regions'][final_abstract_state_index] \
                                    and j in self.params['peripheries'][start_abstract_state_index]:
                                b_i += self.params['transition_matrix'][policy[i_R_index], i_R, j] \
                                       * self.params['reward_matrix'][policy[i_R_index], i_R, j]
                            if j in self.params['regions'][start_abstract_state_index]:
                                b_i += self.params['transition_matrix'][policy[i_R_index], i_R, j] \
                                       * self.params['reward_matrix'][policy[i_R_index], i_R, j]
                        b.append(b_i)
                    b = np.array(b).astype(float)

                    A = np.zeros((size, size))
                    for i in range(size):
                        for j in range(size):
                            action = policy[i]
                            i_global_index = self.params['regions'][start_abstract_state_index][i]
                            j_global_index = self.params['regions'][start_abstract_state_index][j]
                            A[i, j] = self.params['transition_matrix'][action, i_global_index, j_global_index]

                    A = np.eye(size) - A

                    phi = np.linalg.inv(A.T @ A) @ A.T @ b

                    proba = 0

                    for index, state_region in enumerate(self.params['regions'][start_abstract_state_index]):
                        if state_region in self.params['boundaries'][start_abstract_state_index]:
                            proba += phi[index]

                    self.params['abstract_reward_matrix'][abstract_action_index,
                                                          start_abstract_state_index,
                                                          final_abstract_state_index] = proba

        vi = mdptoolbox.mdp.PolicyIteration(self.params['abstract_transition_matrix'],
                                            self.params['abstract_reward_matrix'], 0.999)
        vi.run()
        self.params['abstract_policy'] = vi.policy


def find_region(params, state):
    for i, region in enumerate(params['regions']):
        if state in region:
            return i


def evolve(params, abstract_state, state):
    abstract_action = params['abstract_policy'][abstract_state]
    local_policy = params['local_policies'][abstract_action]
    local_state_index = params['regions'][abstract_state].index(state)
    action = local_policy[local_state_index]
    next_state = np.random.choice(a=list(range(params['state_dim'])), p=params['transition_matrix'][action, state])
    reward = params['reward_matrix'][action, state, next_state]
    return next_state, action, reward


def trajectory(params, state, size=1000):
    states, actions, rewards = [state], [], []
    for _ in range(size):
        abstract_state = find_region(params, state)
        state, action, reward = evolve(params, abstract_state, state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards


def get_policy_matrix(params):
    policy = []
    for state in range(params['state_dim']):
        abstract_state = find_region(params, state)
        policy.append(evolve(params, abstract_state, state)[1])
    return policy
