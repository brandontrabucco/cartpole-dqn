###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

WEIGHTS_EXTENSION = "_weights"
BIASES_EXTENSION = "_biases"
NUMBER_EXTENSION = (lambda n: "_" + str(n))