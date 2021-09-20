import time

import numpy as np
import tensorflow as tf
from numpy import random

from agents import SupervisedAgent
from construct import Construct, Environment, PathGenerator, Wall, LaserEnvironment
from tf_agents.environments import tf_py_environment

tf.keras.backend.set_floatx('float32')


def rng_dataset(num_data, max_val):
    rng = random.default_rng()

    x = rng.integers(0, max_val, (num_data, 2))
    y = x[:, 0:2]  # true red laser positions

    return tf.data.Dataset.from_tensor_slices((tf.cast(x, tf.float32), tf.cast(y, tf.float32)))


def generate_random_data(env, num_train=10000, num_valid=100):

    ds_train = rng_dataset(num_train, env.camera_resolution[1])
    ds_train = ds_train.shuffle(num_train).batch(5, drop_remainder=True)  # TODO: batch sizes

    ds_valid = rng_dataset(num_valid, env.camera_resolution[1])
    ds_valid = ds_valid.batch(5, drop_remainder=True)  # TODO: batch sizes
    
    return ds_train, ds_valid


if __name__ == '__main__':
    # construct = Construct(visualize=False)
    construct = LaserEnvironment(visualize=False, speed_restrictions=False)
    # TODO: this env works but we need to check if we don't have some problem with training the net
    # TODO: try to run with tf_agents
    # construct.run()
    env = Environment()
    pth_gen = PathGenerator()
    wall = Wall(blit=True)

    agent = SupervisedAgent(env, construct, hidden_shape=500)

    ds_train, ds_valid = generate_random_data(env)

    agent.train(ds_train, ds_valid=ds_valid, num_epochs=3, lr=0.001)

    inp_x, inp_y = pth_gen.ellipse(scale=0.5, circle=True, return_angles=True)
    time_step = construct.reset()
    for ix, iy in zip(inp_x, inp_y):

        pred_angles = agent.predict(time_step.observation)

        time_step = construct.step(pred_angles)

        grn_pos = time_step.observation[0:2]
        red_pos = time_step.observation[2:]

        wall.update(red_pos, grn_pos)
        time.sleep(0.01)


# TODO: make reward function based on distance + efficienty of movement + punishment for not moving
# TODO: enforce boundaries for agent to not go out of bounds of the working area (maybe also implement bigger punishment for doing so)
# TODO: make parralelizable for multiagent training
# TODO: make framework for input type switch (either raw image data or laser positions)
# TODO: implement training with regards to speed_restrictions=True
# TODO: batch sizes