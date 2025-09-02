
import irsim
import numpy as np

env = irsim.make('robot_circle.yaml') # initialize the environment with the configuration file

for i in range(300): # run the simulation for 300 steps

   # MOVE MULTIPLE ROBOTS
    actions = [np.array([1.0, 0.5]), np.array([0.5, 0.5]), np.array([0.2, 1.0]), np.array([1.5, 1.0]), np.array([0.7, 0.9])]
    env.step(actions, action_id=[0, 1, 2, 3, 4, 5])  # Move robots 0 and 1

    #MOVE 1 ROBOT
    #env.step(np.array([0.5, 0.5]))  # 1.0 m/s forward, 0.5 rad/s turn
    
    env.render() # render the environment

    if env.done(): break # check if the simulation is done
        
env.end() # close the environment