from abc import ABC

import numpy as np
import tensorflow as tf

from tensorflow import keras

from construct import Construct, Servo


class SimpleNN(keras.Model, ABC):
    """

    """

    def __init__(self, hidden_shape, inp_shape=4, out_shape=2, name="simple-nn", **kwargs):
        super(SimpleNN, self).__init__(name=name, **kwargs)

        self.l1 = keras.layers.Dense(inp_shape)
        self.l2 = keras.layers.Dense(hidden_shape)
        self.l3 = keras.layers.Dense(out_shape)

    @staticmethod
    def float_to_angle_rescaler(x, low=20, high=160):
        """ TODO: don't access protected variables"""
        center = float(high + low)/2
        return center + x*float(center - low)

    def call(self, inputs, **kwargs):
        x = keras.activations.relu(self.l1(inputs))
        x = keras.activations.relu(self.l2(x))
        x = keras.activations.tanh(self.l3(x))
        return tf.cast(tf.round(self.float_to_angle_rescaler(x)), tf.int32)  # values between (-90, 90)


class Agent:

    def __init__(self, env, red_laser=None, green_laser=None, hidden_shape=20, lr=0.001):
        self.env = env
        self.construct = Construct((red_laser, green_laser))

        rng = np.random.default_rng()

        self.net = SimpleNN(hidden_shape)  # base shape TODO: change to be more versatile during init with more models
        self.opt = keras.optimizers.Adam(lr)
        self.loss_metric = keras.metrics.Mean()
        self.valid_loss_metric = keras.metrics.Mean()

    @staticmethod
    def cost(red_batch, grn_batch):
        """ euclidean distance cost function """
        return tf.sqrt(tf.square(red_batch[:, 0] - grn_batch[:, 0]) + tf.square(red_batch[:, 1] - grn_batch[:, 1]))

    def train(self, ds_train, ds_valid=None, num_epochs=10):
        for ep in range(num_epochs):
            for step, (red_grn_batch_train, grn_batch_train) in enumerate(ds_train):
                with tf.GradientTape() as tape:
                    red_batch_angle_pred = self.net(red_grn_batch_train)  # red laser angle prediction

                    done_red = False
                    print("step")
                    while not done_red:
                        # so far works online TODO: allow parrallel calculations for batches
                        done_red = self.construct.step(red_batch_angle_pred[0, 0], red_batch_angle_pred[0, 1])

                    red_batch_pred = tf.expand_dims(self.construct.red_pos, 0)

                    red_batch = tf.cast(red_batch_pred, tf.float32)
                    grn_batch = tf.cast(grn_batch_train, tf.float32)

                    loss = self.cost(red_batch, grn_batch)  # [batch_size, 2]

                grads = tape.gradient(loss, self.net.trainable_variables)
                print(grads)  # no grads calculated TODO: solve!
                self.opt.apply_gradients(zip(grads, self.net.trainable_variables))

                self.loss_metric(loss)

                if step % 1 == 0:
                    print(f"ep: {ep} | step: {step} | mean_loss = {self.loss_metric.result()}")

            if ds_valid and ep % 2 == 0:
                for red_grn_batch_valid, grn_batch_valid in ds_valid:
                    red_batch_servo_angles = self.net(red_grn_batch_valid)

                    loss = self.cost(red_batch_pred, grn_batch_valid)

                    self.valid_loss_metric(loss)

                print(f"ep: {ep} | validation mean_loss = {self.valid_loss_metric.result()}")

    def predict(self, inp):
        pass