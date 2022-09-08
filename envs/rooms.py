import numpy as np
import itertools
import mdptoolbox


class Rooms:
    def __init__(self):
        self.regions = [[] for _ in range(4)]
        self.S = 100
        self.A = 4
        self.P = np.zeros((self.A, self.S, self.S))
        self.R = -1 * np.ones((self.A, self.S, self.S))

        self.size = 10
        self.build_reward_matrix()
        self.build_transition_matrix()
        mdptoolbox.util.check(self.P, self.R)
        self.make_regions()

    @staticmethod
    def coordinates_to_state_index(coordinates):
        """
        Transforms a couple of coordinates into the index of the state.
        """
        return (9 - coordinates[1]) * 10 + coordinates[0]

    @staticmethod
    def state_index_to_coordinates(s):
        """
        Transforms the index of the state into a couple of coordinates.
        """
        return s % 10, 9 - (s // 10)

    def build_reward_matrix(self):
        """
        Set the finish reward to 0.
        """
        for a in range(self.A):
            for s in range(self.S):
                self.R[a, 2, s] = 0

    def build_transition_matrix(self):
        """
        Set the transitions probabilities according to 10.1007/978-3-642-27645-3_9 example of four rooms
        """
        for a in range(self.A):
            for x in range(self.size):
                for y in range(self.size):
                    # North :
                    if (a == 0) and (0 <= y <= 3 or 5 <= y <= 8 or ((x, y) == (2, 4)) or ((x, y) == (7, 4))):
                        s = self.coordinates_to_state_index((x, y))
                        sp = self.coordinates_to_state_index((x, y + 1))
                        self.P[a, s, sp] = 0.8
                        self.P[a, s, s] = 0.2
                    elif a == 0:
                        s = self.coordinates_to_state_index((x, y))
                        self.P[a, s, s] = 1
                    # South
                    if (a == 1) and (1 <= y <= 4 or 6 <= y <= 9 or ((x, y) == (2, 5)) or ((x, y) == (7, 5))) \
                            and ((x, y) != (2, 9)):
                        s = self.coordinates_to_state_index((x, y))
                        sp = self.coordinates_to_state_index((x, y - 1))
                        self.P[a, s, sp] = 0.8
                        self.P[a, s, s] = 0.2
                    elif a == 1:
                        s = self.coordinates_to_state_index((x, y))
                        self.P[a, s, s] = 1
                    # East
                    if (a == 2) and (0 <= x <= 3 or 5 <= x <= 8 or ((x, y) == (4, 2)) or ((x, y) == (4, 7))) \
                            and ((x, y) != (2, 9)):
                        s = self.coordinates_to_state_index((x, y))
                        sp = self.coordinates_to_state_index((x + 1, y))
                        self.P[a, s, sp] = 0.8
                        self.P[a, s, s] = 0.2
                    elif a == 2:
                        s = self.coordinates_to_state_index((x, y))
                        self.P[a, s, s] = 1
                    # West
                    if (a == 3) and (1 <= x <= 4 or 6 <= x <= 9 or ((x, y) == (5, 2)) or ((x, y) == (5, 7))) \
                            and ((x, y) != (2, 9)):
                        s = self.coordinates_to_state_index((x, y))
                        sp = self.coordinates_to_state_index((x - 1, y))
                        self.P[a, s, sp] = 0.8
                        self.P[a, s, s] = 0.2
                    elif a == 3:
                        s = self.coordinates_to_state_index((x, y))
                        self.P[a, s, s] = 1.

    def make_regions(self):
        """
        Divide the state space into four equal rooms.
        """
        for x in range(10):
            for y in range(10):
                if x <= 4:
                    if y <= 4:
                        self.regions[2].append(self.coordinates_to_state_index((x, y)))
                    else:
                        self.regions[0].append(self.coordinates_to_state_index((x, y)))
                else:
                    if y <= 4:
                        self.regions[3].append(self.coordinates_to_state_index((x, y)))
                    else:
                        self.regions[1].append(self.coordinates_to_state_index((x, y)))

        for elem in self.regions:
            elem.sort()

        # Check if the regions constitute a partition of the state space
        assert(sorted(list(itertools.chain.from_iterable(self.regions))) == list(range(self.S)))