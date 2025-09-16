import irsim

# Stuff seems to not work yet propperly yet, the agent seems to want to
# go towards the goal rather than stay on the circle.
# My theories is that either the agents turn radius is messing with it partly,
# there are too many states, which gets too confusing for the agent or my reward
# system still doesn't work properly
# I could try to reduce the amount of actions down to two; going away from the goal
# and going towards it.

# I found out that i had forgotten to update the old_distance value, so i wasn't 
# updating the Q table propperly

# I still think i should reduce the complexity of the environment (amount of states and actions)

# Reduced states down to 50 and actions down to 10, seems to work better now, still doesn't follow the circle that well tho
# So maybe try and tweak the reward system more

for i in range(1):
    env = irsim.make(save_ani=False, full=False)
    env.load_behavior("custom_behavior_methods")

    for _i in range(1000):
        env.step()
        env.render(0.01)

        if env.done():
            break

    env.end(1)