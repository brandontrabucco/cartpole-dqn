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
    
    def random_replay(self):
        return self.tf_environment_utils.sample_randomly(
            self.env, 
            self.tf_environment_args().replay_capacity)

    