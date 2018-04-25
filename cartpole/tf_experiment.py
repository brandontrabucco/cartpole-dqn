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
from datetime import datetime
import matplotlib.pyplot as plt

class TFExperiment(object):

    def __init__(self):
        self.register = TFRegisterArgs()
        self.register("--num_epochs", int, 5)
        self.register("--iterations_per_epoch", int, 100)
        self.register("--logs_per_epoch", int, 10)
        self.register("--test_epoch", int, 0)
        self.register("--use_policy", bool, False)
        self.register("--render_env", bool, False)
        self.tf_environment = TFEnvironment()
        self.tf_agent = TFAgent()

    def train(self):
        args = self.register.parse_args()
        self.tf_agent.construct_parameters()

        loss_checkpoints = []
        for x in range(args.num_epochs):
            self.tf_agent.reset()
            current_loss = 0

            if args.use_policy:
                s, a, r, ns = self.tf_environment.simulate_policy(
                    lambda x: self.tf_agent.test(x).numpy()[0],
                    render=args.render_env)
            else:
                s, a, r, ns = self.tf_environment.simulate_random(
                    render=args.render_env)

            for i in range(args.iterations_per_epoch):
                current_loss = self.tf_agent.train(s, a, r, ns).numpy().sum()

                if i % (args.iterations_per_epoch // args.logs_per_epoch) == 0:
                    print(
                        datetime.now(),
                        "Epoch: %d" % x,
                        "Iteration: %d" % (i + args.iterations_per_epoch * x), 
                        "Loss: %1.2f" % current_loss)

            loss_checkpoints += [(x, current_loss)]
            self.tf_agent.save_parameters(x)

        plt.plot(*zip(*loss_checkpoints), "r-")
        plt.xlabel("Training Epoch (%d Iterations)" % args.iterations_per_epoch)
        plt.ylabel("L2 Loss")
        plt.title("DQN Training Loss")
        plt.grid(True)
        plt.savefig(
            "plots/" 
            + datetime.now().strftime("%Y_%B_%d_%H_%M_%S") 
            + "_training_loss.png")
        plt.close()

        best_epoch, best_loss = min(
            loss_checkpoints, 
            key=(lambda x: x[1]))
        print(
            datetime.now(), 
            "Best Epoch: %d" % best_epoch,
            "Loss: %1.2f" % best_loss)
        return best_epoch, best_loss

    def test(self):
        args = self.register.parse_args()
        
        self.tf_agent.load_parameters(args.test_epoch)
        if args.use_policy:
            self.tf_environment.simulate_policy(
                lambda x: self.tf_agent.test(x).numpy()[0],
                render=args.render_env)
        else:
            self.tf_environment.simulate_random(
                render=args.render_env)