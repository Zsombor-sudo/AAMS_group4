class Network:
    agents = {}
    
    #REGISTER THEM IN ORDER IN TERMS OF IDS!!!
    def register(self, agent):
        self.agents[agent.id] = agent
    
    def send(self, sender_id, targets, message):
        for target in targets:
            self.agents[target].receive_msg(sender_id,message)
        
    
