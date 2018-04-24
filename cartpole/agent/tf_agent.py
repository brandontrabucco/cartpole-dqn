###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.agent.tf_agent_args import TFAgentArgs
from cartpole.agent.tf_agent_utils import TFAgentUtils
from cartpole.agent import WEIGHTS_EXTENSION
from cartpole.agent import BIASES_EXTENSION
from cartpole.agent import NUMBER_EXTENSION
from cartpole.agent import INPUT_SIZE
from cartpole.agent import ACTION_SIZE


class TFAgent(object):

    def __init__(
            self):
        self.tf_agent_args = TFAgentArgs()
        self.tf_agent_utils = TFAgentUtils()

    def construct_parameters(
            self):
        args = self.tf_agent_args()

        self.weights = [self.tf_agent_utils.initialize_weights_cpu(
            ("layer" + NUMBER_EXTENSION(i) + WEIGHTS_EXTENSION),
            [args.controller_breadth, args.controller_breadth],
            decay_factor=0.99
        ) for i in range(1, args.controller_depth)]

        self.weights = [self.tf_agent_utils.initialize_weights_cpu(
            ("layer" + NUMBER_EXTENSION(0) + WEIGHTS_EXTENSION),
            [INPUT_SIZE, args.controller_breadth],
            decay_factor=0.99
        )] + self.weights

        self.weights = self.weights + [self.tf_agent_utils.initialize_weights_cpu(
            ("layer" + NUMBER_EXTENSION(args.controller_depth) + WEIGHTS_EXTENSION),
            [args.controller_breadth],
            decay_factor=0.99
        )]

        self.biases = [self.tf_agent_utils.initialize_biases_cpu(
            ("layer" + NUMBER_EXTENSION(i) + BIASES_EXTENSION),
            [args.controller_breadth]
        ) for i in range(1, args.controller_depth)]

        self.biases = [self.tf_agent_utils.initialize_biases_cpu(
            ("layer" + NUMBER_EXTENSION(0) + BIASES_EXTENSION),
            [args.controller_breadth]
        )] + self.biases

        self.biases = self.biases + [self.tf_agent_utils.initialize_biases_cpu(
            ("layer" + NUMBER_EXTENSION(args.controller_depth) + BIASES_EXTENSION),
            [1]
        )]

    def reset(self):
        args = self.tf_agent_args()
        self.optimizer = self.tf_agent_utils.optimizer(
            args.learning_rate,
            args.decay_rate,
            args.decay_steps)

    def get_parameters(self):
        return self.weights + self.biases
        
    def inference(
            self,
            x_batch):
        args = self.tf_agent_args()

        for i, (w, b) in enumerate(zip(
                self.weights, 
                self.biases)):
            x_batch = self.tf_agent_utils.layer_forward(
                x_batch, w, b, last=(i == args.controller_depth))
        return x_batch

    def train(
            self,
            state,
            action,
            reward,
            next_state):
        args = self.tf_agent_args()

        (state_batch,
            action_batch,
            reward_batch,
            next_state_batch) = self.tf_agent_utils.generate_batch(
                (state, action, reward, next_state),
                args.batch_size,
                args.replay_capacity)
        
        def tape_buffer():
            q_estimate = self.inference(
                self.tf_agent_utils.concat(state_batch, action_batch))
            q_actual = reward_batch + args.discount_factor * self.tf_agent_utils.max_action(
                self.inference, 
                next_state_batch, 
                ACTION_SIZE,
                args.batch_size)
            return self.tf_agent_utils.l2_loss(
                q_estimate, 
                q_actual)

        self.tf_agent_utils.minimize(
            self.optimizer, 
            tape_buffer, 
            self.get_parameters())
        return tape_buffer()

    def test(
            self,
            current_state):
        args = self.tf_agent_args()

        return self.tf_agent_utils.argmax_action(
            self.inference, 
            current_state, 
            ACTION_SIZE,
            args.replay_capacity)
        