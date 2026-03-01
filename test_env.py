from env import GridWorld
import random

env = GridWorld()

state = env.reset()
done = False

while not done:
    action = random.randint(0, 3)
    state, reward, done = env.step(action)
    env.render()
    print("Reward:", reward)
    print("------")