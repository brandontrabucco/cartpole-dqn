###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.tf_experiment import TFExperiment

if __name__ == "__main__":
    net = TFExperiment()
    net.test()