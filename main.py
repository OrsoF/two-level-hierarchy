from env import rooms, tandem

solver_name = 'mdptoolbox'
env_name = '4rooms'
algorithm_name = 'hierarchy'

assert solver in ['mdptoolbox', 'marmot']
assert env in ['4rooms', 'tandemqueue']
assert algorithm in ['hierarchy', 'policy_iteration', 'value_iteration']

