###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.args.tf_register_args import TFRegisterArgs

class TFEnvironmentArgs(object):

    def __init__(self):
        self.register = TFRegisterArgs()
        self.register("--replay_capacity", int, 10000)
        self.register("--batch_size", int, 100)
        self.register("--action_size", int, 2)
        self.register("--state_size", int, 4)
        self.register("--env_name", str, "CartPole-v0")

    def __call__(self):
        return self.register.parse_args()