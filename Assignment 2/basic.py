import irsim

for i in range(1):
    env = irsim.make(save_ani=False, full=False, display=True)
    # env = irsim.make(save_ani=False, full=True, display=True)
    env.load_behavior("custom_basic_circle")
    env.load_behavior("custom_follower_behaviour")

    for _i in range(1000):
        env.step()

        # for rob in env.robot_list:
        #     if rob.fov_detect_object(env.robot):
        #         detected_robots = rob.get_fov_detected_objects()
        #         for detRob in detected_robots:
                    # print(f'Agent {rob.id} detected agent {detRob.id}')

        # obs = env.object_factory.create_object(
        #     obj_type= 'obstacle', 
        #     number= 1, 
        #     state= [[_i, _i, 0]], 
        #     shape= {'name': 'circle', 'radius': 0.2},
        #     color= 'red'
        # )
        # env.add_objects(obs)

        # # Patch for older/newer IR-Sim mismatch
        # for o in obs:
        #     if not hasattr(o, 'plot_attr_list') and hasattr(o, 'plot_patch_list'):
        #         o.plot_attr_list = o.plot_patch_list
        #         o.show_trail = o.plot_trail

        #         # Patch missing attributes
        #         if not hasattr(o, 'trail_freq'):
        #             o.trail_freq = 1        # must be ≥1 to avoid modulo by zero
        #         if not hasattr(o, 'show_trail'):
        #             o.show_trail = False    # disables trail drawing
        #         if not hasattr(o, 'trail_color'):
        #             o.trail_color = getattr(o, 'color', 'black')
        #         if not hasattr(o, 'trail_data'):
        #             o.trail_data = []

        #         # Assign axes from the environment
        #         if not hasattr(o, 'ax'):
        #             o.ax = env._env_plot.ax  # the main Matplotlib axes

        #         # Initialize plotting-related attributes
        #         if not hasattr(o, 'plot_kwargs'):
        #             o.plot_kwargs = {}
        #         if not hasattr(o, 'original_vertices'):
        #             o.original_vertices = getattr(o, 'vertices', None)  # Some objects have .vertices
                
        #         # Sensor / FOV attributes
        #         if not hasattr(o, 'show_sensor'):
        #             o.show_sensor = False
        #         if not hasattr(o, 'sensor_range'):
        #             o.sensor_range = 0
        #         if not hasattr(o, 'sensor_fov'):
        #             o.sensor_fov = 0

        # env.render(interval= 0.01, mode= "all")

        if env.done():
            break

    env.end(1)
    