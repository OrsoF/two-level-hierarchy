from envs.rooms import Rooms
from envs.tandem import Tandem
from solvers.mdptoolbox_solver import solve
from solvers.hierarchy import Regions
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--solver', type=str,
                    help='exact or hierarchical')
parser.add_argument('--env', type=str,
                    help='rooms or tandem_queue')
parser.add_argument('--method', type=str,
                    help='policy_iteration or value_iteration')
parser.add_argument('--gamma', type=float,
                    help='Discount rate')

args = parser.parse_args()

available_solvers = ['exact', 'hierarchical']
envs = ['rooms', 'tandem_queue']
methods = ['value_iteration', 'policy_iteration']

if args.solver is None:
    solver = 'exact'
else:
    solver = args.solver

if args.env is None:
    env = 'rooms'
else:
    env = args.env

if args.method is None:
    method = 'value_iteration'
else:
    method = args.method

if args.gamma is None:
    gamma = 0.99
else:
    gamma = args.gamma

assert solver in available_solvers
assert env in envs
assert method in methods
assert 0 < gamma <= 1

if env == 'rooms':
    env = Rooms()
else:
    env = Tandem()

if solver == 'exact':
    vi = solve(env.P, env.R, gamma, method)
else:
    hierarchy = Regions(env, -5, method)
