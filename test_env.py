from env import GridWorld
import random

env = GridWorld(grid_size=8, dynamic_obstacles=True, obstacle_prob=0.15)

state = env.reset()
done = False

while not done:
    action = random.randint(0, 3)
    state, reward, done = env.step(action)
    env.render()
    print("Reward:", reward)
    print("------")