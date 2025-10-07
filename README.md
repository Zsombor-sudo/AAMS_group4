# Multi-Agent Systems - Group 4

## Summary of layered approach for collision avoidance

To improve the robot's behavior, an additional layer was introduced to handle collision avoidance. This enhancement expanded the original state space by implementing a binary indicator for detecting threats, wheter a robot is within the safety distance, which indicates a threat, or not, thereby doubling the number of states. The action space was also extended with a new "stop" action, which allows the robot to remain stationary when it detects a collision threat. A new reward function was designed to penalize proximity to other robots, unnecessary stopping, and crashes, while still encouraging progress towards the goal of circular movement. By applying a layered approach, it becomes possible to retain the original navigation behavior while adding a simple safety mechanism that activates only when necessary.

In safe conditions, the robot behaves similarly to the original behavior, navigating around the target in a circular movement. However, when another robot comes within the safety distance, the robot modifies its policy by stopping. This could be expanded by introducing some sort of turning-away behavior to avoid the collisions completely. The behavior develops naturally from the Q-learning updates and the enhanced rewards. When structuring such a system in layers, the robot benefits from a modular implementation where the collision avoidance layer acts as an override mechanism, enhancing robustness in multi-agent scenarios without disrupting the original core objectives.

## Collision testing

The evaluation for the collision avoidance was a bit skewed due to the fact that the robots are initiated in random positions. This means that in some runs, the robots start very close to each other, which leads to immediate collisions on the first few steps. However, evaluating the performance over multiple runs gives a rough indication of the effectiveness of the collision avoidance layer using the stopping mechanism. Here are some results from testing the collision avoidance layer:

| Run | # of Collisions | Collision Rate (%) |
|-----|-----------------|--------------------|
