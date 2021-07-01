import tensorflow as tf
from numpy import random

from agents import Agent
from construct import Environment, Construct

if __name__ == '__main__':
    # construct = Construct(visualize=True)
    # construct.run()
    env = Environment()

    rng = random.default_rng()

    train_x = rng.integers(60, 121, (1000, 4), dtype=int)
    train_y = train_x[:, 0:2]  # red laser positions

    ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds_train = ds_train.batch(1)

    agent = Agent(env)

    agent.train(ds_train)
