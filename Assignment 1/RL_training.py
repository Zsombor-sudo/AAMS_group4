import irsim
import numpy as np
from pathlib import Path

# file_path = Path('q_table.csv') # When executing the file while in the Assignemnt 1 folder
file_path = Path('Assignment 1/q_table.csv') # When executing the file while in the Assignemnt 1 parent folder

# If it doesn't exist, create a Q table for 500 states (distance to center in 0.1 increments) and 314 actions (angle to center between 0 and 3.14 in 0.01 increments)
if not file_path.exists():
    q_table = np.zeros([500, 314])
else: # Otherwise load the Q table:
    q_table = np.load(file_path, delimiter=',', dtype=float)

# Save the Q table:
np.savetxt(file_path, q_table, delimiter=',', fmt='%f')