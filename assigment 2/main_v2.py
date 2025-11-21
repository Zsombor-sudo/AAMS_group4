
import irsim
import numpy as np
import matplotlib.pyplot as plt
from irsim.util.util import WrapToPi, relative_position
from irsim.world.object_base import ObjectBase
from network import Network
from network import BULLY_MSG
import types
import threading
import time

N_NODES = 5
network = Network()
electionRun = True
# Receive message from other agents
def receiveMsg(self, sender, distance, winnerId):
    #print("Message from: "+str(sender))
    if sender == -1:
        self.leaderID = winnerId
        return
    if self.id == 0:
        sendVictory(winnerId)
    else:
        thisAgentDist = calculateDistanceToGoal(self,goal_pos)
        if thisAgentDist < distance:
            sendToNextAgent(self.id,self.id,thisAgentDist)
        else:
            sendToNextAgent(self.id,winnerId,distance)



#send a message to other agents
def sendMsg(self, targets, message):
    network.send(self.id,targets,message)

#helper methods for election message handling
def sendElection(id):
    #send election to higher ranks
    network.send(id,range(id+1,N_NODES),BULLY_MSG.ELECTION)

def sendVictory(id):
    network.send(-1,range(0,N_NODES-1),-1,id)
    print("Leader: "+str(id))

def waitForVictory(agent):
    time.sleep(2)
    if agent.leaderID == -1:
        sendVictory(agent,agent.id)

def sendToNextAgent(senderId,winnerId,distance):
    if senderId < N_NODES-1:
        network.send(senderId,[senderId+1],distance,winnerId)
    else:
        network.send(senderId,[0],distance,winnerId)

def start():
    electionRun = True

def calculateDistanceToGoal(agent,goal_pos):
    dist, _ = relative_position(agent.state[:2],goal_pos)
    return dist

#run method for agents leader election
def bullyRun(agent):
    while(True):
        if agent.id == 0:
            dist = calculateDistanceToGoal(agent,goal_pos)
            sendToNextAgent(0,0,dist)
        
        electionRun = False
        while(electionRun==False):
            pass

def leaderElectByRing(goal_pos):
    agent_0 = network.agents[0]
    dist = calculateDistanceToGoal(agent_0,goal_pos)
    sendToNextAgent(0,0,dist)
    return agent_0.leaderID

env = irsim.make('basic.yaml')

for agent in env.robot_list:
    #add message methods to the agents
    agent.receiveMsg = types.MethodType(receiveMsg, agent)
    agent.sendMsg = types.MethodType(sendMsg, agent)
    #agent.distanceToGoal = 10*agent.id+5
    network.register(agent)

#need to register all before starting threads
#for agent in env.robot_list:
#    agent.t = threading.Thread(target=bullyRun,args=(agent,))
#    agent.t.start()


leader = env.robot_list[0]
followers = env.robot_list[1:]

leader_max_forward = 1  # leader max forward speed
follower_max_forward = 0.9
max_turn = 1            # max angular speed

# --- Pheromone parameters ---
world_size = (25, 25)
pheromone_map = np.zeros(world_size)
deposit_rate = 1.0
evaporation_rate = 0.3

# --- Functions ---
def elect_new_leader_closest_to_goal(current_leader: ObjectBase, followers: list[ObjectBase]):
    """
    Sample a new random goal and pick the robot (leader or follower) closest to it as new leader.
    Resets pheromone map. Returns (new_leader, followers).
    """
    # Reset pheromone map so followers track new leader only
    global pheromone_map
    pheromone_map = np.zeros_like(pheromone_map)

    # Sample new random goal
    new_goal = np.random.uniform(0, 25, size=(2, 1))
    
    # Find closest robot to new goal
    candidates = [current_leader] + followers
    closest_idx = leaderElectByRing(new_goal)
    new_leader = candidates[closest_idx]

    # If the new leader was a follower, adjust lists
    if new_leader is not current_leader:
        followers.remove(new_leader)
        followers.append(current_leader)
        new_leader.color = 'r'

        # Demote current leader to follower
        current_leader.color = 'g'
        current_leader.set_goal([-1,-1,0])  # no specific goal for follower
    # Else leader stays leader; followers unchanged

    # Assign goal to new leader
    new_leader.set_goal(new_goal.flatten().tolist() + [0])
    print(f"New leader (closest to goal) elected: Robot ID {new_leader.id}")
    return new_leader, followers

def deposit_pheromone(pos):
    """
    Leader leaves pheromone at current position.
    FORMAL:  τ_ij ← τ_ij + Δτ_ij
    
    """
    x, y = int(round(pos[0,0])), int(round(pos[1,0]))
    if 0 <= x < world_size[0] and 0 <= y < world_size[1]:
        pheromone_map[x, y] += deposit_rate
    

def evaporate_pheromone():
    
    """
    Global pheromone evaporation.
    FORMAL: τ_ij ← (1 - ρ) * τ_ij
    where ρ = evaporation_rate
    
    """
    global pheromone_map
    pheromone_map *= (1 - evaporation_rate)


def move_follower(follower):
    """
    Follower moves based on pheromone concentrations.
    Slows down if other robots are detected in its FOV.
    """
    # Current position and orientation
    pos = follower.state[:2]
    theta = follower.state[2,0]
    x, y = int(round(pos[0,0])), int(round(pos[1,0]))
    max_r = 25  # maximum expansion radius
    best_cell = None
    best_val = -1.0

    # --- Expand search for pheromone peak ---
    for r in range(1, max_r + 1):
        found = False
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx = x + dx
                cy = y + dy
                if 0 <= cx < world_size[1] and 0 <= cy < world_size[0]:
                    val = pheromone_map[cx, cy]
                    if val > best_val and val > 0:
                        best_val = val
                        best_cell = (cx, cy)
                        found = True
        if found:
            break


    # --- Check neighbors in FOV for slowing down ---
    neighbor_list = follower.get_fov_detected_objects()  # robots in FOV
    slowdown_factor = 2.0  # default (full speed)

    if neighbor_list:
        min_distance = min(
            np.linalg.norm(neighbor.state[:2] - pos) for neighbor in neighbor_list
        )
        # Slowdown factor decreases linearly with distance
        # Closer neighbors -> slower speed, minimum 0.01
        slowdown_factor = max(0.01, min_distance / 1.0)  

  
    target = np.array([[best_cell[0]],[best_cell[1]]], dtype=float)
    _, target_angle = relative_position(pos, target)

    # --- Steering towards pheromone peak ---
    angle_diff = WrapToPi(target_angle - theta)
    v_angular = np.clip(angle_diff, -max_turn, max_turn)

    # --- Repulsion (linear term) ---
    v_forward = follower_max_forward * slowdown_factor

    follower.step(np.array([[v_forward], [v_angular]]))


for step in range(5000):
    
    # Check if leader reached its goal
    leader.check_arrive_status()
    if leader.arrive_flag:
        print(f"Leader reached goal at step {step}. Electing new leader.")
        # DEFINE NEW LEADER GOAL
        new_goal = np.random.uniform(0, 25, size=(2,))

        leader.set_goal([new_goal[0], new_goal[1], 0])

        #USING SELECT LEADER FUNCTION
        leader, followers = elect_new_leader_closest_to_goal(leader, followers)

    pos = leader.state[:2]
    theta = leader.state[2,0]
    goal_pos = leader.goal[0:2]
    dist, target_angle = relative_position(pos, goal_pos)
    angle_diff = WrapToPi(target_angle - theta)

    if dist > leader.goal_threshold:
        v_forward = min(leader_max_forward, float(dist))
        v_angular = np.clip(angle_diff, -max_turn, max_turn)
        leader.step(np.array([[v_forward],[v_angular]]))
    else:
        leader.step(np.zeros((2,1)))

   
    # --- Pheromone update ---
    deposit_pheromone(pos) # τ_ij ← τ_ij + Δτ_ij
    evaporate_pheromone()  # τ_ij ← (1 - ρ) * τ_ij

    # --- Follower behavior ---
    for f in followers:
        move_follower(f)


    env.render()
    
    #if env.done(): break # check if the simulation is done

    ax = plt.gca()
   
    ax.grid(True)

env.end()


