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
        self.global_step = tf.contrib.eager.Variable(
            0, 
            trainable=False)
        self.decay_shift = tf.contrib.eager.Variable(
            0, 
            trainable=False)

    def initialize_weights_cpu(
            self, 
            name,
            shape=None,
            value=None):
        if value is not None:
            init = tf.constant(value)
            return tf.get_variable(
                name,
                initializer=init,
                dtype=tf.float32)
        elif shape is not None:
            init = tf.truncated_normal_initializer(
                stddev=0.01)
            return tf.get_variable(
                name,
                shape=shape,
                initializer=init,
                dtype=tf.float32)

    def initialize_biases_cpu(
            self,
            name,
            shape=None,
            value=None):
        if value is not None:
            init = tf.constant(value)
            return tf.get_variable(
                name,
                initializer=init,
                dtype=tf.float32)
        elif shape is not None:
            init = tf.constant_initializer(1.0)
            return tf.get_variable(
                name,
                shape=shape,
                initializer=init,
                dtype=tf.float32)

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

    def create_optimizer(
            self,
            learning_rate,
            decay_rate,
            decay_steps):
        self.global_optimizer = tf.train.AdamOptimizer(
            tf.train.exponential_decay(
                learning_rate,
                self.global_step - self.decay_shift,
                decay_steps,
                decay_rate))

    def reset_decay(self):
        tf.assign(self.decay_shift, self.global_step)

    def minimize(
            self, 
            loss,
            parameters):
        return self.global_optimizer.minimize(
            loss, 
            global_step=self.global_step,
            var_list=parameters)

    def concat(
            self, 
            state, 
            action,
            action_size):
        action = tf.one_hot(action, action_size)
        return tf.concat([
            state, 
            action], axis=-1)

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
        return state, next_action

    def max_action(
            self, 
            inference, 
            next_state, 
            action_size,
            batch_size):
        return tf.stop_gradient(tf.reduce_max(
            inference(
                *self.expand_actions(
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
                *self.expand_actions(
                    current_state, 
                    action_size, 
                    batch_size)), 
                axis=1))