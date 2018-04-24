###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.args.tf_register_args import TFRegisterArgs

class TFAgentArgs(object):

    def __init__(self):
        self.register = TFRegisterArgs()
        self.register("--learning_rate", float, 0.001)
        self.register("--decay_rate", float, 0.99)
        self.register("--decay_steps", float, 100)
        self.register("--discount_factor", float, 0.99)
        self.register("--controller_depth", int, 4)
        self.register("--controller_breadth", int, 16)

    def __call__(self):
        return self.register.parse_args()