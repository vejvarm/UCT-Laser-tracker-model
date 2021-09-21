from abc import ABC
from typing import Tuple

import numpy as np
import tensorflow as tf

from tensorflow import keras


class SimpleNN(keras.Model, ABC):
    """

    """

    def __init__(self, hidden_shape, inp_size=4, out_size=2, name="simple-nn", **kwargs):
        super(SimpleNN, self).__init__(name=name, **kwargs)

        self.l1 = keras.layers.Dense(inp_size)
        self.l2 = keras.layers.Dense(hidden_shape)
        self.l3 = keras.layers.Dense(out_size)

        self.relu = keras.activations.relu
        self.tanh = keras.activations.tanh

        self.build(input_shape=(None, inp_size))

        self.summary()

    @staticmethod
    def float_to_angle_rescaler(x, low=60, high=120):
        """ TODO: don't access protected variables"""
        center = float(high + low)/2
        return center + x*float(center - low)

    def call(self, inputs, **kwargs):
        x = self.relu(self.l1(inputs))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        x = self.tanh(x)
        return self.float_to_angle_rescaler(x)  # values between (-90, 90) not rounded


class Agent:

    def __init__(self):
        pass


class SupervisedAgent:

    def __init__(self, env, construct, hidden_shape=20):
        self.env = env
        self.construct = construct

        self.net = SimpleNN(hidden_shape)  # base shape TODO: change to be more versatile during init with more models
        self.opt = keras.optimizers.Adam
        self.loss = tf.losses.mean_squared_error
        self.loss_metric = keras.metrics.Mean()
        self.valid_loss_metric = keras.metrics.Mean()

    def train(self, ds_train, ds_valid=None, num_epochs=10, lr=0.001):
        optimizer = self.opt(lr)
        grn_pos = [self.construct.green_pos]*5  # TODO: batch sizes
        for ep in range(num_epochs):
            for step, (red_pos, grn_pos_target) in enumerate(ds_train):
                with tf.GradientTape() as tape:
                    grn_red_pos = tf.concat((grn_pos, red_pos), axis=1)
                    # TODO: normalize more globally (during data creation/feeding into net?)
                    grn_red_pos = (grn_red_pos - self.env.camera_resolution[1]/2)/self.env.camera_resolution[1]
                    grn_batch_angle_pred = self.net(grn_red_pos, training=True)  # green laser angle prediction
                    # print(red_batch_angle_pred)

                    # DONE: MAKE DIFFERENTIABLE! or some other solution
                    # angle to meter fun
                    grn_pos = tf.map_fn(lambda x: tf.cast(self.env.angle_to_pixel(x), tf.float32), grn_batch_angle_pred)

                    # print(red_batch_angle_pred, red_pos_batch)

                    # print(self.net.trainable_variables)

                    loss = self.env.cost(tf.concat((grn_pos, red_pos), -1))
                    # print(grn_pos, grn_pos_target)
                    # loss = self.loss(grn_pos, grn_pos_target)  # [batch_size, 2]

                    # tape.watch(loss)
                    # print(loss.shape)

                    # print([var.shape for var in tape.watched_variables()])

                grads = tape.gradient(loss, self.net.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 10.0)
                #
                optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

                self.loss_metric(loss)

                if step % 100 == 0:
                    print(f"ep: {ep} | step: {step} | mean_loss = {self.loss_metric.result()} | grad = {tf.reduce_mean([tf.reduce_mean(g) for g in grads])}")

            if ds_valid and ep % 2 == 0:
                for red_pos_valid, grn_target_valid in ds_valid:
                    grn_pos_valid = [self.construct.green_pos] * len(red_pos_valid)
                    grn_red_pos_valid = tf.concat((grn_pos_valid, red_pos_valid), axis=1)
                    # TODO: normalize more globally
                    grn_red_pos_valid = (grn_red_pos_valid - self.env.camera_resolution[1] / 2) / self.env.camera_resolution[1]
                    grn_batch_servo_angles = self.net(grn_red_pos_valid, training=False)

                    grn_batch_pos = tf.map_fn(lambda x: tf.cast(self.env.angle_to_pixel(x), tf.float32),
                                              grn_batch_servo_angles)

                    loss = self.env.cost(tf.concat((grn_batch_pos, red_pos_valid), -1))

                    self.valid_loss_metric(loss)

                print(f"ep: {ep} | validation mean_loss = {self.valid_loss_metric.result()}")

    def predict(self, observation):
        """

        :param observation: Array[int] current pixel positions of green and red laser
        :return predicted_angles: Tuple[int] new angle position of the green laser servos
        """
        inputs = tf.expand_dims(observation, 0)
        # TODO: normalize more globally (during data creation/feeding into net?)
        inputs = (inputs - self.env.camera_resolution[1] / 2) / self.env.camera_resolution[1]
        predicted_angles = self.net(inputs, training=False)

        return predicted_angles
