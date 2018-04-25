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

    def sample_off_policy(
            self, 
            env,
            capacity,
            render=False):
        return self.sample_on_policy(
            env, 
            capacity, 
            lambda x: env.action_space.sample(), 
            render=render)

    def sample_on_policy(
            self, 
            env, 
            capacity, 
            policy,
            render=False):
        iterations = 0
        state = []
        action = []
        reward = []
        next_state = []

        while iterations < capacity:
            initial_state = env.reset()
            for i in range(capacity):

                if render:
                    env.render()
                directed_action = policy(
                    initial_state[np.newaxis, :].astype(
                            np.float32))
                (result_state, 
                    result_reward, 
                    done, 
                    info) = env.step(
                        directed_action)

                iterations += 1
                state += [initial_state]
                action += [directed_action]
                reward += [result_reward]
                next_state += [result_state]

                if done or iterations == capacity:
                    break
                else:
                    initial_state = result_state

        state = np.vstack(state).astype(np.float32)
        action = np.squeeze(np.vstack(action))
        reward = np.squeeze(np.vstack(
            reward).astype(np.float32))
        next_state = np.vstack(
            next_state).astype(np.float32)
        return (state, 
            action, 
            reward, 
            next_state)