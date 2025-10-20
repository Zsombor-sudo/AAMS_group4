import irsim

for i in range(1):
    # env = irsim.make(save_ani=False, full=False, display=False)
    env = irsim.make(save_ani=False, full=False, display=True)
    env.load_behavior("custom_basic_circle")
    env.load_behavior("custom_RL_circle")

    for _i in range(500):
        env.step()
        env.render(0.01)

        if env.done():
            break

    env.end(1)
    