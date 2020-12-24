import sys
import os

machines = [
    'madison',
    'montgomery',
    'montpelier',
    'nashville',
    'oklahoma-city',
    'olympia',
    'phoenix',
    'pierre',
    'providence',
    'raleigh',
    'richmond',
    'sacramento',
    'saint-paul',
    'salem',
    'salt-lake-city',
    'santa-fe',
    'springfield',
    'tallahassee',
    'topeka',
    'moscow',
    'albany',
    'annapolis',
    'atlanta',
    'augusta',
    'austin',
    'baton-rouge',
    'bismarck',
    'boise',
    'boston',
    'carson-city',
    'charleston',
    'cheyenne',
    'columbia',
    'columbus-oh',
    'concord',
    'denver',
    'des-moines',
    'dover',
    'frankfort',
    'harrisburg',
    'hartford',
    'helena',
    'honolulu',
    'indianapolis',
    'jackson',
    'jefferson-city',
    'juneau',
    'lansing'
    'lincoln',
    'little-rock'
]

no_of_machines = len(machines)

if len(sys.argv) != 3:
    print('Usage: python3 distributed_runner.py <world_size> <path_to_python_script>')
    sys.exit(0)

world_size = int(sys.argv[1])

if world_size > no_of_machines:
    print("World Size should be less than " + str(no_of_machines))
    sys.exit(0)

selected_machines = machines[:world_size]
python_script = sys.argv[2]

for i in range(len(selected_machines)):
    command = 'ssh ' + selected_machines[i] + ' "python3 ' + python_script + ' ' + str(i) + ' ' + str(world_size) + '" &'
    print('Spawning: ' + command)
    os.system(command)
