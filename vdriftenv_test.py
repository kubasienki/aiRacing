import vdrift_rl 
import gym
#rom vdriftenv import VDriftEnv
#import vdrift_rl

env = env = gym.make('VDrift-v0')
obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    action[0] = -19.0
    obs, reward, done,s, info = env.step(action)
    #if done == True:
    #    break

env.close()
