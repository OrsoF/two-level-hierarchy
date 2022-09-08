from envs.rooms import Rooms
from solvers.mdptoolbox_solver import solve

env = Rooms()
vi = solve(env.P, env.R)
