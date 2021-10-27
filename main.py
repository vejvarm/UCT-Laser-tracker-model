import time

import numpy as np
import tensorflow as tf
from numpy import random

from agents import SupervisedAgent
from environment import LaserTracker, Wall
from transformations import Transformations, PathGenerator
from tf_agents.environments import tf_py_environment, tf_environment
from tf_agents.environments import utils

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.specs import BoundedArraySpec, tensor_spec
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver

from tensorflow.keras import metrics

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


def run_supervised():
    construct = LaserTracker(visualize=False, speed_restrictions=False)
    # DONE: this env works but we need to check if we don't have some problem with training the net
    env = Transformations()
    pth_gen = PathGenerator()
    wall = Wall(blit=True)

    agent = SupervisedAgent(env, construct, hidden_shape=500)

    ds_train, ds_valid = generate_random_data(env)

    agent.train(ds_train, ds_valid=ds_valid, num_epochs=10, lr=0.001)

    inp_x, inp_y = pth_gen.ellipse(scale=0.5, circle=True, return_angles=True)
    time_step = construct.reset()
    for ix, iy in zip(inp_x, inp_y):

        pred_angles = agent.predict(time_step.observation)

        time_step = construct.step(pred_angles)

        grn_pos = time_step.observation[0:2]
        red_pos = time_step.observation[2:]

        wall.update(red_pos, grn_pos)
        time.sleep(0.01)


# 04 METRICS and EVALUATION
def compute_avg_return(environment, policy, num_episodes=10):
    """ Average return (sum of rewards during episode) over 「num_episodes」 episodes

    :param environment:
    :param policy:
    :param num_episodes:
    :return:
    """

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)  # policy determines next action
            time_step = environment.step(action_step.action)  # apply action to env and get new state
            episode_return += time_step.reward  # add reward to episode return sum
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def run_rl_agent():
    pass


if __name__ == '__main__':
    # 00 HYPERPARAMS
    num_runs = 10
    num_iterations = 50  # @param {type:"integer"}
    collect_episodes_per_iteration = 10  # @param {type:"integer"}
    replay_buffer_capacity = 500  # @param {type:"integer"}

    fc_layer_params = (256, 128, 64)

    learning_rate = 1e-5  # @param {type:"number"}
    log_interval = 25  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 50  # @param {type:"integer"}

    # 01 THE ENVIRONMENT
    train_env_py = LaserTracker(visualize=False, speed_restrictions=False)
    eval_env_py = LaserTracker(visualize=False, speed_restrictions=False)

    # convert to Tensorflow compatible environments
    train_env = tf_py_environment.TFPyEnvironment(train_env_py)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)
#    utils.validate_py_environment(train_env, episodes=5)

    # print(train_env.reset())

    # print(isinstance(train_env, tf_environment.TFEnvironment))
    # print("TimeStep Specs:", train_env.time_step_spec())
    # print("Action Specs:", train_env.action_spec())

    # 02 THE AGENT
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),  # BoundedArraySpec(shape=(4, ), dtype=dtype('float32'), name='observation', minimum=[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], maximum=[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38])
        train_env.action_spec(),       # BoundedArraySpec(shape=(2, ), dtype=dtype('float32'), name='action', minimum=0, maximum=180)
        fc_layer_params=fc_layer_params
    )

    # create optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),  # what is this? (oh it's just state of given time step)
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter
    )

    tf_agent.initialize()  # initialize RL Reinforce agent
    print(tf_agent.collect_data_spec)

    # 03 P0LICY
    eval_policy = tf_agent.policy  # used for evaluation/deployment/production
    collect_policy = tf_agent.collect_policy  # used for data collection

    # 04 METRICS (see function compute_avg_return @top-of-file)
    # 05 REPLAY BUFFER
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(tf_agent.collect_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=replay_buffer_capacity,
                                                                   )

    # Add an observer that adds to the replay buffer:
    replay_observer = [replay_buffer.add_batch]

    print(replay_observer)

    driver = dynamic_episode_driver.DynamicEpisodeDriver(train_env,
                                                         tf_agent.collect_policy,
                                                         observers=replay_observer,
                                                         num_episodes=collect_episodes_per_iteration)
    # DONE: This is never gonna end, because we have to define WHEN is the END of the EPISODE

    loss_mean = metrics.Mean()
    for _ in range(num_runs):

        final_time_step, policy_state = driver.run()  # DONE: there is a problem with shapes!

        # Read the replay buffer as a Dataset,
        # read batches of 4 elements, each with 50 timesteps:
        dataset = replay_buffer.as_dataset(
            sample_batch_size=5,
            num_steps=num_iterations)  # must be same as steps_per_ep in LaserTracker env!

        loss_mean.reset_state()
        for _ in range(num_iterations):
            iterator = iter(dataset)
            trajectories, _ = next(iterator)
            # print(trajectories)
            loss = tf_agent.train(experience=trajectories)
            loss_mean.update_state(loss.loss)
        print(loss_mean.result())


# DONE: REINFORCE requires full episodes to compute losses.
# TODO: How not to instantly explode into 0,180 angle bounds?
# TODO: make reward function based on distance + efficienty of movement + punishment for not moving
# TODO: enforce boundaries for agent to not go out of bounds of the working area (maybe also implement bigger punishment for doing so)
# TODO: make parralelizable for multiagent training
# TODO: make framework for input type switch (either raw image data or laser positions)
# TODO: implement training with regards to speed_restrictions=True
# TODO: batch sizes