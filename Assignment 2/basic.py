import irsim

for i in range(1):
    env = irsim.make(save_ani=False, full=False, display=True)
    env.load_behavior("custom_basic_circle")
    env.load_behavior("custom_follower_behaviour")

    for _i in range(1000):
        env.step()

        # for rob in env.robot_list:
        #     if rob.fov_detect_object(env.robot):
        #         detected_robots = rob.get_fov_detected_objects()
        #         for detRob in detected_robots:
                    # print(f'Agent {rob.id} detected agent {detRob.id}')

        env.render(0.01)

        if env.done():
            break

    env.end(1)
    