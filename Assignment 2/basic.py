import math
import time
import irsim
import numpy as np
from network import Network
from network import BULLY_MSG
import types
import threading
# MAX SPEED
V_MAX = 1.0
PI = 3.1415926
N_NODES = 5
network = Network()

# Receive message from other agents
def receiveMsg(self, sender, message, arg=0):
    #print("Message from: "+str(sender))
    match message:
        case BULLY_MSG.ELECTION:
            if sender<self.id and self.canSendAnything==1:
                network.send(self.id,[sender],BULLY_MSG.ALIVE)
                sendElection(self.id)
        case BULLY_MSG.ALIVE:
            if sender>self.id:
               waitForVictory(self)
            else:
                self.isAlive = 1
        case BULLY_MSG.VICTORY:
            #accept the leader
            self.leaderID = sender
            self.stopThread = 1


#send a message to other agents
def sendMsg(self, targets, message):
    network.send(self.id,targets,message)

def sendElection(id):
    #send election to higher ranks
    network.send(id,range(id+1,N_NODES),BULLY_MSG.ELECTION)

def sendVictory(self,id):
    if self.stopThread == 0 and self.leaderID == -1:    
        network.send(id,range(0,id-1),BULLY_MSG.VICTORY)
        print("Leader: "+str(id))

def waitForVictory(agent):
    time.sleep(2)
    if agent.leaderID == -1:
        sendVictory(agent,agent.id)


def bullyRun(agent):
    #print("agent here: "+str(agent.id))
    
    while(True):
        if agent.id != N_NODES-1:
            agent.isAlive = 0
            if agent.canSendAnything == 1:
                sendElection(agent.id)
            #time out on 1 seconds for not receiving alive msg
            time.sleep(1)
            if agent.stopThread == 1:
                break
            if not agent.isAlive:
                sendVictory(agent,agent.id)
                break
        else:
            sendVictory(agent,agent.id)
            break

        if agent.stopThread == 1:
            break
        #restart when one is dead
            
    
    



    

for i in range(1):
    env = irsim.make('basic.yaml')
    #env.load_behavior("custom_behavior_methods")

    
    for agent in env.robot_list:
        agent.receiveMsg = types.MethodType(receiveMsg, agent)
        agent.sendMsg = types.MethodType(sendMsg, agent)
        agent.leaderID = -1
        agent.canSendAnything = 1
        agent.isAlive = 0
        agent.kill = -1
        agent.stopThread = 0
        
        network.register(agent)

    #need to register all before starting threads
    for agent in env.robot_list:
        agent.t = threading.Thread(target=bullyRun,args=(agent,))
        agent.t.start()
    #avoid threads blocking eachother 
    

    time.sleep(2)
    print("kill")
    network.killAgent(4)
    

    for agent in env.robot_list:
        agent.t.join()
    agents = env.robot_list

    #Example usage:
    #agents[3].send_msg([0,1,2],"Hello!!")

    # grab the 5 proxies you defined
    proxies = [o for o in env.obstacle_list if getattr(o, "name", "").startswith("proxy_r")]
    proxies.sort(key=lambda o: o.name)  # ensure order proxy_r1..proxy_r5

    env.render(interval= 0.01, mode= "all")
    for _i in range(1000):
        env.step()
        # mirror robot poses into proxy obstacles

        # create a rectangular obstacle dynamically
        sy=12
        y_pos=8+sy/100
        obs = env.object_factory.create_object(
        obj_type= 'obstacle', 
        number= 1, 
        state= [[12,sy, 0]], 
        shape= {'name': 'circle', 'radius': 0.2},
        color= 'red'
        )
        #leaderState.state[0]
        env.add_objects(obs)
        
        for r, p in zip(agents, proxies):
            p.state[0] = r.state[0] + 0.1
            p.state[1] = r.state[1] + 0.1
            p.state[2] = r.state[2]  # [x, y, theta]
        actions = []
        for robot in env.robot_list[:-1]:
            #if obs.fov_detect_object(env.robot):
            robot.sensor_step()
            minDistIdx = np.argmin(robot.sensors[0].range_data)
            angleToRobot = robot.sensors[0].angle_list[minDistIdx]
            actions.append(np.array([V_MAX,angleToRobot]))
            print(min(robot.sensors[0].range_data))
            #print(robot.sensors[0].angle_list)
            #print(f'The robot is in the FOV of the {obs.name}. The parameters of this obstacle are: state [x, y, theta]: {obs.state.flatten()}, velocity [linear, angular]: {obs.velocity.flatten()}, fov in radian: {obs.fov}.')
            #print(irsim.world.sensors.lidar2d.Lidar2D.get_scan(self))
                
        leaderangle = 0
        match _i:
            case v if 0 <= v <= 250:
                leaderangle = 0
            case v if 251 <= v <= 500:
                leaderangle = PI/2
            case v if 501 <= v <= 750:
                leaderangle = PI
            case v if 751 <= v <= 1000:
                leaderangle = -PI/2
            case _:
                leaderangle = 0
        leaderState = env.robot_list[0]
        print(float(leaderState.state[0]))
        obs = env.object_factory.create_object(
        obj_type= 'obstacle', 
        number= 1, 
        state= [[float(leaderState.state[0]),float(leaderState.state[1])-0.5, 0]], 
        shape= {'name': 'circle', 'radius': 0.2},
        color= 'red'
        )
        
        #leaderState.state[0]
        env.add_objects(obs)
        actions.append(np.array([V_MAX,leaderangle]))

        env.step(actions)

        env.render(0.01)

        if env.done():
            break

    env.end(1)

#DRAFT FOR SOLVE LIDAR2D-BUG SCRAPED:
# (optional but recommended) avoid self-hits: set lidar range_min > robot radius
    #for r in robots:
    #    for s in r.sensors:
    #        if getattr(s, "name", "") == "lidar2d":
    #            if hasattr(s, "range_min"):
    #                s.range_min = max(getattr(s, "range_min", 0.0), 0.15)
    #obs1 = env.object_factory.create_object(
    #        name="box1",
    #        obj_type='obstacle',
    #        shape={'name': 'rectangle', 'length': 1.0, 'width': 0.5},
    #        state=[5.0, 5.0, 0.0],   # x, y, theta
    #    )[0]
    #factory = irsim.world.object_factory()
    
    #####OLD######
    #obs1 = env.object_factory.create_object(
    #        obj_type='obstacle',
    #        number=1,
    #        state=[[5, 5, 0.0]]
    #    )
    #env.add_objects(obs1)
    #env.build_tree()
    ###############
    ###NEW######
    #for rob in env.robot_list:
    #    if rob.fov_detect_object(env.robot):
    #        detected_robots = rob.get_fov_detected_objects()
    #        for detRob in detected_robots:
    #            print(f'Agent {rob.id} detected agent {detRob.id}')


    

    # Patch for older/newer IR-Sim mismatch
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

    #env.render(interval= 0.01, mode= "all")
    