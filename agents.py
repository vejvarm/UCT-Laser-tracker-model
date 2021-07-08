from abc import ABC

import numpy as np
import tensorflow as tf

from tensorflow import keras

from construct import Construct, Servo, Wall


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
    def float_to_angle_rescaler(x, low=20, high=160):
        """ TODO: don't access protected variables"""
        center = float(high + low)/2
        return center + x*float(center - low)

    def call(self, inputs, **kwargs):
        x = self.relu(self.l1(inputs))
        x = self.relu(self.l2(x))
        x = self.tanh(self.l3(x))
        return self.float_to_angle_rescaler(x)  # values between (-90, 90) not rounded


class Agent:

    def __init__(self, env, red_laser=None, green_laser=None, hidden_shape=20):
        self.env = env
        self.construct = Construct((red_laser, green_laser))

        self.net = SimpleNN(hidden_shape)  # base shape TODO: change to be more versatile during init with more models
        self.opt = keras.optimizers.Adam
        self.loss = tf.losses.mean_squared_error
        self.loss_metric = keras.metrics.Mean()
        self.valid_loss_metric = keras.metrics.Mean()

        self.wall = Wall(blit=True)

    @staticmethod
    def cost(red_batch, grn_batch):
        """ euclidean distance cost function """
        return tf.sqrt(tf.square(red_batch[:, 0] - grn_batch[:, 0]) + tf.square(red_batch[:, 1] - grn_batch[:, 1]))

    def train(self, ds_train, ds_valid=None, num_epochs=10, lr=0.001):
        optimizer = self.opt(lr)
        for ep in range(num_epochs):
            for step, (red_grn_batch_train, grn_batch_train) in enumerate(ds_train):
                with tf.GradientTape() as tape:
                    red_batch_angle_pred = self.net(red_grn_batch_train, training=True)  # red laser angle prediction

                    # DONE: MAKE DIFFERENTIABLE! or some other solution
                    # angle to meter fun
                    red_pos_batch = tf.map_fn(lambda x: tf.cast(self.env.angle_to_pixel(x), tf.float32),
                                              red_batch_angle_pred)

                    # print(red_batch_angle_pred, red_pos_batch)

                    # print(self.net.trainable_variables)

                    # loss = self.loss(grn_batch_train, red_batch_angle_pred)  # [batch_size, 2]
                    loss = self.cost(red_pos_batch, grn_batch_train)

                    # tape.watch(loss)
                    # print(loss.shape)

                    # print([var.shape for var in tape.watched_variables()])

                grads = tape.gradient(loss, self.net.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

                self.loss_metric(loss)

                if step % 100 == 0:
                    print(f"ep: {ep} | step: {step} | mean_loss = {self.loss_metric.result()}")

            if ds_valid and ep % 2 == 0:
                for red_grn_batch_valid, grn_batch_valid in ds_valid:
                    red_batch_servo_angles = self.net(red_grn_batch_valid, training=False)

                    red_batch_pos = tf.map_fn(lambda x: tf.cast(self.env.angle_to_pixel(x), tf.float32),
                                              red_batch_servo_angles)

                    loss = self.cost(red_batch_pos, grn_batch_valid)

                    self.valid_loss_metric(loss)

                print(f"ep: {ep} | validation mean_loss = {self.valid_loss_metric.result()}")

    def predict(self, inp):
        done_red = False
        i_red = 0
        while not done_red and i_red < 10:
            red_pos = self.construct.step(inp[0], inp[1])
            i_red += 1
        input_angles = tf.expand_dims(tf.convert_to_tensor((*self.construct.red_pos, *self.construct.green_pos), dtype=tf.float32), 0)
        predicted_angles = self.net(input_angles, training=False)

        done_green = False
        i_green = 0
        while not done_green and i_green < 10:
            green_pos = self.construct.step(predicted_angles[0, 0], predicted_angles[0, 1])
            i_green += 1

        # TODO: return done argument form "construct.step!"

        self.wall.update(red_pos, green_pos)
