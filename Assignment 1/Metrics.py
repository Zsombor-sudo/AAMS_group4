import matplotlib.pyplot as plt
import numpy as np
import csv

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
        self.speed = [0] * self.numberOfAgents
        self.angular = [0] * self.numberOfAgents

        #open the csv file to save measurements
        self.fRad = open("Assignment 1/RadiusErrors.csv", "a", newline="")
        self.fAng = open("Assignment 1/AngleErrors.csv", "a", newline="")
        self.fSpeed = open("Assignment 1/Speed.csv", "a", newline="")

        self.writerRad = csv.writer(self.fRad)
        self.writerAng = csv.writer(self.fAng)
        self.writerSpeed = csv.writer(self.fSpeed)
        
    def setCircleRadius(self, radius):
        self.circle_radius = radius
        
    def update(self, robotid, distance):
        #printing error
        self.distance[robotid] = distance
        self.accDistance[robotid] += abs(distance - self.circle_radius)
        self.iterations[robotid] += 1
    
    def updateSpeed(self, robotid, speed):
        self.speed[robotid] = speed
    
    def updateAngular(self, robotid, angular):
        self.angular[robotid] = angular

        

    def print(self,robotid):
        #printing error
        self.errLabelText[robotid].set_text(f" {round(abs(self.distance[robotid] - self.circle_radius),2)}  ||  {round(self.accDistance[robotid], 2)}  ||  {round(self.accDistance[robotid]/self.iterations[robotid], 2)}")

        #condition if all agents are on same iteration, then save record
        if not len(set(self.iterations)) == 1:
            return

        #save to file
        self.writerRad.writerow(round(abs(dist - self.circle_radius),2) for dist in self.distance)
        self.fRad.flush()

        #angular
        self.writerAng.writerow(round(abs(angular - np.pi/2),2) for angular in self.angular)
        self.fAng.flush()

        #speed
        self.writerSpeed.writerow(round(abs(speed),2) for speed in self.speed)
        self.fSpeed.flush()

