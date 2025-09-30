class Metrics:
    def __init__():
        errLabelObj = plt.gcf()
        errLabelText = [errLabelObj.text(0.15, 0.75, "Placeholder ", ha="left", va="top"),
                errLabelObj.text(0.15, 0.80, "Placeholder ", ha="left", va="top"),
                errLabelObj.text(0.15, 0.85, "Placeholder ", ha="left", va="top"),
                errLabelObj.text(0.15, 0.90, "Placeholder ", ha="left", va="top")]
        #explaination label
        plt.gcf().text(0.15,0.95,"Current error [m] | accumulated error [m] | average error per iteration [m/iter]", ha="left", va="top")
    
    def update():
        #printing error
        accDistance += abs(distance_float - circle_radius)
        iterations += 1    
    def print(robotid):
        #printing error
        errLabelText[robotid].set_text(f" {abs(distance - circle_radius)}  ||  {round(accDistance, 2)}  ||  {round(accDistance/iterations, 2)}")

