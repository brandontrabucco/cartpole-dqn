###########################
# Reset working directory #
###########################
from os import chdir
chdir("G:\\My Drive\\Academic\\Research\\Cartpole Simulation")
###########################
# Cartpole Package....... #
###########################

from cartpole.agent import LOSS_COLLECTION
import tensorflow as tf

class TFAgentUtils(object):

    def __init__(self):
        tf.enable_eager_execution()
        self.global_step = None

    def initialize_weights_cpu(
            self, 
            name,
            shape,
            standard_deviation=0.01,
            decay_factor=None):
        weights = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)
        return weights

    def initialize_biases_cpu(
            self,
            name,
            shape):
        biases = tf.get_variable(
            name,
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)
        return biases

    def generate_batch(
            self, 
            x_inputs,
            batch_size,
            capacity):
        indices = tf.random_uniform(
            [batch_size], 
            maxval=capacity, 
            dtype=tf.int32)
        return [tf.gather(x, indices) for x in x_inputs]

    def layer_forward(
            self,
            x, 
            w, 
            b, 
            last=False):
        x = tf.tensordot(x, w, 1) + b
        if not last:
            x = tf.nn.relu(x)
        return x

    def l2_loss(
            self,
            prediction,
            labels):
        l2_norm = tf.nn.l2_loss(
                labels - prediction)
        return l2_norm

    def optimizer(
            self,
            learning_rate,
            decay_rate,
            decay_steps):
        self.global_step = tf.contrib.eager.Variable(
            0, 
            trainable=False)
        return tf.train.AdamOptimizer(
            tf.train.exponential_decay(
                learning_rate,
                self.global_step,
                decay_steps,
                decay_rate))

    def minimize(
            self, 
            optimizer,
            loss,
            parameters):
        return optimizer.minimize(
            loss, 
            global_step=self.global_step,
            var_list=parameters)

    def reset(self):
        tf.assign(self.global_step, 0)

    def concat(
            self, 
            state, 
            action):
        return tf.concat([
            tf.cast(state, tf.float32), 
            tf.one_hot(tf.squeeze(action), 2)], axis=-1)

    def expand_actions(
            self,  
            state, 
            action_size,
            batch_size):
        state = tf.tile(tf.expand_dims(
            state, 
            axis=1), [1, action_size, 1])
        next_action = tf.tile(tf.reshape(
            tf.range(action_size), 
            [1, action_size]), [batch_size, 1])
        return self.concat(
            state, 
            next_action)

    def max_action(
            self, 
            inference, 
            next_state, 
            action_size,
            batch_size):
        return tf.stop_gradient(tf.reduce_max(
            inference(
                self.expand_actions(
                    next_state, 
                    action_size, 
                    batch_size)), 
                axis=1))

    def argmax_action(
            self, 
            inference, 
            current_state, 
            action_size,
            batch_size):
        return tf.stop_gradient(tf.argmax(
            inference(
                self.expand_actions(
                    current_state, 
                    action_size, 
                    batch_size)), 
                axis=1))
    