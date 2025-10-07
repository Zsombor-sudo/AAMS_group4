# main.py
import argparse
import os

import irsim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train","eval"], default="train")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--render", action="store_true")
parser.add_argument("--qcsv", default="q_table.csv")
args = parser.parse_args()

os.environ["IRSIM_QTABLE_CSV"] = args.qcsv

if args.mode == "eval":
    os.environ["IRSIM_TRAIN"] = "0"      # no learning, no saving
    os.environ.setdefault("IRSIM_EVAL_EPS", "0.0")  # greedy by default
    NUM_EPISODES, STEPS_PER_EP = 1, args.steps
else:
    os.environ["IRSIM_TRAIN"] = "1"
    NUM_EPISODES, STEPS_PER_EP = args.episodes, args.steps

# np.random.seed(42)

env = irsim.make(save_ani=False, full=False, world_name="robot_world.yaml")
env.load_behavior("custom_behavior")

for ep in range(NUM_EPISODES):
    try:
        env.reset()
        for i, robot in enumerate(env.robot_list):
            robot.set_goal([[5, 5, 0]])
            x, y = robot.state[0,0], robot.state[1,0]
            # Make the robot face tangent to the center of the circle
            robot.state[2,0] = np.arctan2(5 - y, 5 - x) + np.pi / 2
            # print(f"Robot {i} starting at ({x:.2f}, {y:.2f}, {robot.state[2,0]:.2f})")
        env.reset_plot()    
    except Exception:
        print("Reset failed, retrying...")
        env = irsim.make(save_ani=False, full=False, world_name="robot_world.yaml")
        env.load_behavior("custom_behavior")

    for t in range(STEPS_PER_EP):
        env.step()
        if args.render: env.render(0.01)
        if env.done(): break

    print(f"{args.mode.title()} episode {ep+1}/{NUM_EPISODES} finished at step {t+1}")

env.end(0)
