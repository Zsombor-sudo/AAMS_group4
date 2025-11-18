from enum import Enum
class BULLY_MSG(Enum):
    ELECTION = 1
    ALIVE = 2
    OK = 3
    VICTORY = 4
class Network:
    agents = {}
    
    #REGISTER THEM IN ORDER IN TERMS OF IDS!!!
    def register(self, agent):
        self.agents[agent.id] = agent
    
    def send(self, sender_id, targets, message):
        if len(targets) < 1:
            return
        for target in targets:
            self.agents[target].receiveMsg(sender_id,message)
    
    def makeLeader(self,id):
        for id in self.agents:
            self.agents[id].leaderID = id
    def killAgent(self,id):
        for id in self.agents:
            self.agents[id].stopThread = 0
            self.agents[id].leaderID = -1
            self.agents[id].kill = id
    
