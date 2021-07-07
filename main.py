import time

import tensorflow as tf
from numpy import random

from agents import Agent
from construct import Environment, PathGenerator

tf.keras.backend.set_floatx('float64')

if __name__ == '__main__':
    # construct = Construct(visualize=True)
    # construct.run()
    env = Environment()
    pth_gen = PathGenerator()

    rng = random.default_rng()

    train_x = rng.integers(0, env.camera_resolution[1], (10000, 4))
    train_y = train_x[:, 0:2]  # true red laser positions

    ds_train = tf.data.Dataset.from_tensor_slices((tf.cast(train_x, tf.float64), tf.cast(train_y, tf.float64)))
    ds_train = ds_train.shuffle(10000).batch(20, drop_remainder=True)

    valid_x = rng.integers(0, env.camera_resolution[1], (100, 4))
    valid_y = valid_x[:, 0:2]  # true red laser positions

    ds_valid = tf.data.Dataset.from_tensor_slices((tf.cast(valid_x, tf.float64), tf.cast(valid_y, tf.float64)))
    ds_valid = ds_valid.batch(5)

    agent = Agent(env, hidden_shape=100)

    inp_x, inp_y = pth_gen.ellipse(scale=0.5, circle=True, return_angles=True)

    agent.train(ds_train, ds_valid=ds_valid, num_epochs=10, lr=0.1)

    for i in range(1000):
        agent.predict((next(inp_x), next(inp_y)))
        time.sleep(0.01)
