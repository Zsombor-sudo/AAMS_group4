import irsim

# Create a q table for 500 states (distance to center in 0.1 increments) and 314 actions (angle to center between 0 and 3.14 in 0.01 increments)
q_table = np.zeros([500, 314]) 