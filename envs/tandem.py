import numpy as np
from solvers.utils import normalize


class Tandem:
    def __init__(self):
        self.lambda_ = 0.6
        self.mu_1 = 0.2
        self.mu_2 = 0.2
        self.B1 = 5
        self.B2 = 5
        self.K1 = 5
        self.K2 = 5
        self.CA = 1
        self.CD = 1
        self.CH = 1
        self.CS = 1
        self.CR = 1

        self.lambda_tilde = self.lambda_ + self.K1 * self.mu_1 + self.K2 * self.mu_2

        self.gamma = 0.99

        dim = np.product([self.B1 + 1, self.B2 + 1, self.K1, self.K2])
        self.S = dim
        self.A = 9

        self.state_encoding = {}
        self.state_decoding = []
        i = 0
        for b1 in range(self.B1 + 1):
            for b2 in range(self.B2 + 1):
                for k1 in range(1, self.K1 + 1):
                    for k2 in range(1, self.K2 + 1):
                        self.state_encoding[(b1, b2, k1, k2)] = i
                        self.state_decoding.append((b1, b2, k1, k2))
                        i += 1

        self.action_encoding = {}
        self.action_decoding = []
        i = 0

        for a1 in range(-1, 2):
            for a2 in range(-1, 2):
                self.action_encoding[(a1, a2)] = i
                self.action_decoding.append((a1, a2))
                i += 1

        self.P = np.zeros((self.A, self.S, self.S), dtype=np.float32)
        self.build_p()
        self.R = np.zeros((self.A, self.S, self.S))
        self.build_r()

        self.space = [(0, self.B1 + 1), (0, self.B2 + 1), (1, self.K1 + 1), (1, self.K2 + 1)]
        self.states = np.zeros(tuple((len(range(*dim)) for dim in self.space)))
        for i, m1 in enumerate(range(*self.space[0])):
            for j, m2 in enumerate(range(*self.space[1])):
                for k, k1 in enumerate(range(*self.space[2])):
                    for m, k2 in enumerate(range(*self.space[3])):
                        self.states[i, j, k, m] = self.state_encoding[(m1, m2, k1, k2)]
        self.regions = []
        self.make_regions()
        self.reformat_regions()

    def n1(self, k):
        return min(max(1, k), self.K1)

    def n2(self, k):
        return min(max(1, k), self.K2)

    def Lambda(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.lambda_ + self.mu_1 * min(m1, self.n1(k1 + a1)) + self.mu_2 * min(m2, self.n2(k2 + a2))

    def c1(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.n1(k1 + a1) * self.CS + m1 * self.CH

    def c2(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.n2(k2 + a2) * self.CS + m2 * self.CH

    def h1(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.CA * int(a1 == 1) + self.CD * int(a1 == -1) + self.lambda_ * self.CR * int(m1 == self.B1 - 1) / (
                self.Lambda(s, a) + self.gamma)

    def h2(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.CA * int(a2 == 1) + self.CD * int(a2 == -1) + min(m1, self.n1(k1 + a1)) * self.mu_1 * self.CR * int(
            m2 == self.B2 - 1) / (self.Lambda(s, a) + self.gamma)

    def s1p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[(min(m1 + 1, self.B1), m2, self.n1(k1 + a1), self.n2(k2 + a2))]

    def s2p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[(max(m1 - 1, 0), min(m2 + 1, self.B2), self.n1(k1 + a1), self.n2(k2 + a2))]

    def s3p(self, s, a):
        m1, m2, k1, k2 = self.state_decoding[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[(m1, max(m2 - 1, 0), self.n1(k1 + a1), self.n2(k2 + a2))]

    def Reward(self, s, a):
        return (self.c1(s, a) + self.c2(s, a)) / (self.Lambda(s, a) + self.gamma) + self.h1(s, a) + self.h2(s, a)

    def Reward_tilde(self, s, a):
        return self.Reward(s, a) * (self.Lambda(s, a) + self.gamma) / (self.lambda_tilde + self.gamma)

    def build_p(self):
        for a in range(self.A):
            for s in range(self.S):
                state, action = self.state_decoding[s], self.action_decoding[a]
                m1, m2, k1, k2 = state
                a1, a2 = action
                if m1 == self.B1 and self.n1(k1 + a1) == a1 and self.n2(k2 + a2) == a2:
                    self.P[a, s, s] += self.lambda_
                else:
                    self.P[a, s, self.s1p(s, a)] += self.lambda_
                    self.P[a, s, self.s2p(s, a)] += self.mu_1 * min(m1, self.n1(k1 + a1))
                    self.P[a, s, self.s3p(s, a)] += self.mu_2 * min(m2, self.n2(k2 + a2))
                    self.P[a, s, s] += self.lambda_tilde - self.Lambda(s, a)

        self.P /= self.lambda_tilde
        self.P = normalize(self.P)

    def build_r(self):
        for a in range(self.A):
            for s in range(self.S):
                self.R[a, s] = self.Reward_tilde(s, a)

    def make_regions(self, axis=0):
        assert isinstance(axis, int) or isinstance(axis, tuple)
        assert isinstance(axis, int) or 2 <= len(axis) < 4

        if isinstance(axis, int):
            self.regions = np.split(self.states, len(range(*self.space[axis])), axis)
            self.regions = [list(np.ndarray.flatten(region)) for region in self.regions]

        elif len(axis) == 2:
            split_1 = np.split(self.states, len(range(*self.space[axis[0]])), axis[0])
            split_2 = [np.split(splitted, len(range(*self.space[axis[1]])), axis[1]) for splitted in split_1]
            self.regions = [list(np.ndarray.flatten(splitted_elem))
                            for splitted_set in split_2
                            for splitted_elem in splitted_set]

        elif len(axis) == 3:
            split_1 = np.split(self.states, len(range(*self.space[axis[0]])), axis[0])
            split_2 = [np.split(splitted, len(range(*self.space[axis[1]])), axis[1])
                       for splitted in split_1]
            split_3 = [np.split(splitted_1, len(range(*self.space[axis[2]])), axis[2])
                       for splitted in split_2
                       for splitted_1 in splitted]
            self.regions = [list(np.ndarray.flatten(splitted_elem))
                            for splitted_set_2 in split_3
                            for splitted_set_1 in splitted_set_2
                            for splitted_elem in splitted_set_1]
        assert sum(len(region) for region in self.regions) == self.S

    def reformat_regions(self):
        self.regions = [[int(state) for state in region] for region in self.regions]
