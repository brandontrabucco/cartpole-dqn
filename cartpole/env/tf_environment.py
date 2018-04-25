###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.env.tf_environment_args import TFEnvironmentArgs
from cartpole.env.tf_environment_utils import TFEnvironmentUtils
import gym

class TFEnvironment(object):

    def __init__(self):
        self.tf_environment_args = TFEnvironmentArgs()
        self.tf_environment_utils = TFEnvironmentUtils()
        self.env = gym.make('CartPole-v0')
    
    def simulate_random(self, render=False):
        args = self.tf_environment_args()
        return self.tf_environment_utils.sample_off_policy(
            self.env, 
            args.replay_capacity,
            render=render)

    def simulate_policy(self, policy, render=False):
        args = self.tf_environment_args()
        return self.tf_environment_utils.sample_on_policy(
            self.env, 
            args.replay_capacity,
            policy,
            render=render)

    