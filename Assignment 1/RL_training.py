import irsim
import numpy as np
from pathlib import Path

## Create/Load Q table
# file_path = Path('q_table.csv') # When executing the file while in the Assignemnt 1 folder
file_path = Path('Assignment 1/q_table.csv') # When executing the file while in the Assignemnt 1 parent folder

# If it doesn't exist, create a Q table for 500 states (distance to center in 0.1 increments) and 314 actions (angle to center between 0 and 3.14 in 0.01 increments)
if not file_path.exists():
    q_table = np.zeros([500, 314])
else: # Otherwise load the Q table:
    q_table = np.load(file_path, delimiter=',', dtype=float)

## Train agent
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Training
env = irsim.make('basic.yaml')
# env = irsim.make(save_ani=False, full=False)
# env.load_behavior("custom_behavior_methods")

trainingLength = 100
for i in range(trainingLength):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0,1) < epsilon:
            action = env.action_spaace.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # exploit learned values

        next_state, reeward, done, info = env.step(action)
        env.render(0.01)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1

env.end(5)

# Save the Q table:
np.savetxt(file_path, q_table, delimiter=',', fmt='%f')