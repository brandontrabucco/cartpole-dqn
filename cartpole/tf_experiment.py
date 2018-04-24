###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.args.tf_register_args import TFRegisterArgs
from cartpole.env.tf_environment import TFEnvironment
from cartpole.agent.tf_agent import TFAgent
import gym

class TFExperiment(object):

    def __init__(self):
        self.register = TFRegisterArgs()
        self.register("--num_epochs", int, 20)
        self.register("--iterations_per_epoch", int, 10000)
        self.register("--logs_per_epoch", int, 10)
        self.tf_environment = TFEnvironment()
        self.tf_agent = TFAgent()

    def train(self):
        args = self.register.parse_args()
        self.tf_agent.construct_parameters()
        for x in range(args.num_epochs):
            self.tf_agent.reset()
            s, a, r, ns = self.tf_environment.random_replay()
            for i in range(args.iterations_per_epoch):
                if i % (args.iterations_per_epoch // args.logs_per_epoch) == 0:
                    print(
                        "Epoch: %d" % x,
                        "Iteration: %d" % (i + 10000* x), 
                        "Loss: %1.2f" % self.tf_agent.train(s, a, r, ns).numpy().sum())

    def test(self):
        self.tf_agent.construct_parameters()
        s, a, r, ns = self.tf_environment.random_replay()
        print(self.tf_agent.test(s).numpy())