###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

import numpy as np

class TFEnvironmentUtils(object):

    def __init__(self):
        pass

    def sample_randomly(
            self, 
            env,
            capacity):
        iterations = 0
        state = []
        action = []
        reward = []
        next_state = []

        while iterations < capacity:
            initial_state = env.reset()
            for i in range(capacity):

                random_action = env.action_space.sample()
                (result_state, 
                    result_reward, 
                    done, 
                    info) = env.step(
                    random_action)
                iterations += 1
                state += [initial_state]
                action += [random_action]
                reward += [result_reward]
                next_state += [result_state]

                if done or iterations == capacity:
                    break
                else:
                    initial_state = result_state

        state = np.vstack(state).astype(np.float32)
        action = np.vstack(action)
        reward = np.vstack(reward).astype(np.float32)
        next_state = np.vstack(next_state).astype(np.float32)
        return (state, 
            action, 
            reward, 
            next_state)
