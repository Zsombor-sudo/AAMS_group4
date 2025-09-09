import irsim

for i in range(10):
    env = irsim.make(save_ani=False, full=False)
    env.load_behavior("custom_behavior_methods")

    for _i in range(1000):
        env.step()
        env.render(0.01)

        if env.done():
            break

    env.end(1)