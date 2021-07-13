import time

import tensorflow as tf
from numpy import random

from agents import Agent
from construct import Construct, Environment, PathGenerator, Wall

tf.keras.backend.set_floatx('float32')


def generate_random_data(env, num_train=10000, num_valid=100):
    rng = random.default_rng()

    train_x = rng.integers(0, env.camera_resolution[1], (10000, 4))
    train_y = train_x[:, 0:2]  # true red laser positions

    ds_train = tf.data.Dataset.from_tensor_slices((tf.cast(train_x, tf.float32), tf.cast(train_y, tf.float32)))
    ds_train = ds_train.shuffle(10000).batch(20, drop_remainder=True)

    valid_x = rng.integers(0, env.camera_resolution[1], (100, 4))
    valid_y = valid_x[:, 0:2]  # true red laser positions

    ds_valid = tf.data.Dataset.from_tensor_slices((tf.cast(valid_x, tf.float32), tf.cast(valid_y, tf.float32)))
    ds_valid = ds_valid.batch(5)
    
    return ds_train, ds_valid


if __name__ == '__main__':
    construct = Construct(visualize=False)
    # construct.run()
    env = Environment()
    pth_gen = PathGenerator()
    wall = Wall(blit=True)

    agent = Agent(env, construct, hidden_shape=100)
    
    ds_train, ds_valid = generate_random_data(env)

    inp_x, inp_y = pth_gen.ellipse(scale=0.5, circle=True, return_angles=True)

    agent.train(ds_train, ds_valid=ds_valid, num_epochs=1, lr=0.0000001)

    grn_pos = construct.green_pos
    for ix, iy in zip(inp_x, inp_y):
        _, red_pos = construct.step(ix, iy, speed_restrictions=False)

        pred_angles = agent.predict(red_pos, grn_pos)

        _, grn_pos = construct.step(pred_angles[0, 0], pred_angles[0, 1], speed_restrictions=False)

        wall.update(red_pos, grn_pos)
        time.sleep(0.01)



