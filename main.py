from envs.rooms import Rooms
from envs.tandem import Tandem
from solvers.mdptoolbox_solver import solve
from solvers.hierarchy import Hierarchy
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--solver', type=str,
                    help='exact or hierarchical')
parser.add_argument('--env', type=str,
                    help='rooms or tandem_queue')

args = parser.parse_args()

available_solvers = ['exact', 'hierarchical']
envs = ['rooms', 'tandem_queue']

if args.solver is None:
    solver = 'exact'
else:
    solver = args.solver

if args.env is None:
    env = 'rooms'
else:
    env = args.env

assert solver in available_solvers
assert env in envs

if env == 'rooms':
    env = Rooms()
else:
    env = Tandem()

if solver == 'exact':
    vi = solve(env.P, env.R)
else:
    hierarchy = Hierarchy(env)


