from envs.rooms import Rooms
from envs.tandem import Tandem
from solvers.mdptoolbox_solver import solve
from solvers.hierarchy import Regions

available_solvers = ['exact', 'hierarchical']
envs = ['rooms', 'tandem_queue']
methods = ['value_iteration', 'policy_iteration']

solver = 'hierarchical'
env = 'rooms'
method = 'policy_iteration'

assert solver in available_solvers
assert env in envs
assert method in methods

if env == 'rooms':
    env = Rooms()
else:
    env = Tandem()

if solver == 'exact':
    vi = solve(env.P, env.R, method)
else:
    hierarchy = Regions(env, -5, method)
    hierarchy.test_solution()
