import matplotlib.pyplot as plt

class Metrics:
    numberOfAgents = 4
    errLabelObj = plt.gcf()
    errLabelText = [None] * numberOfAgents
    for i in range(numberOfAgents):
        errLabelText[i] = errLabelObj.text(0.15, 0.90-0.05*i, "Placeholder ", ha="left", va="top")
    #explaination label
    plt.gcf().text(0.15,0.95,"Current error [m] | accumulated error [m] | average error per iteration [m/iter]", ha="left", va="top")

    def __init__(self, circle_radius):
        self.accDistance = [0] * self.numberOfAgents
        self.iterations = [1] * self.numberOfAgents
        self.circle_radius = circle_radius
        self.distance = [0] * self.numberOfAgents
    def setCircleRadius(self, radius):
        self.circle_radius = radius
    def update(self, robotid, distance):
        #printing error
        self.distance[robotid] = distance
        self.accDistance[robotid] += abs(distance - self.circle_radius)
        self.iterations[robotid] += 1    
    def print(self,robotid):
        #printing error
        self.errLabelText[robotid].set_text(f" {round(abs(self.distance[robotid] - self.circle_radius),2)}  ||  {round(self.accDistance[robotid], 2)}  ||  {round(self.accDistance[robotid]/self.iterations[robotid], 2)}")

