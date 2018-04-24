###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.args.tf_static_args import TFStaticArgs

class TFRegisterArgs(object):

    def __init__(
            self):
        pass

    def __call__(
            self, 
            sname, 
            stype, 
            sdefault):
        TFStaticArgs.add_static_arg(
            sname, 
            stype, 
            sdefault)

    def parse_args(self):
        return TFStaticArgs.parse_all_args()