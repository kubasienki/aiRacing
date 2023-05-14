from vdriftenv import VDriftEnv

env = VDriftEnv()
obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done == True:
        break

env.close()