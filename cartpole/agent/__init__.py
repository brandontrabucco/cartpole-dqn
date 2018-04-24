###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

LOSS_COLLECTION = "loss_collection"
PARAMETER_COLLECTION = "parameter_collection"
WEIGHTS_EXTENSION = "_weights"
BIASES_EXTENSION = "_biases"
NUMBER_EXTENSION = (lambda n: "_" + str(n))

STATE_SIZE = 4
ACTION_SIZE = 2
INPUT_SIZE = STATE_SIZE + ACTION_SIZE