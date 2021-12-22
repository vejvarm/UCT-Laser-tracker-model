import time

import tensorflow as tf
from numpy import random
from tensorflow.keras import metrics
from tf_agents.agents.ddpg import ddpg_agent, critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

from agents import SupervisedAgent
from environment import LaserTracker, Wall
from transformations import Transformations, PathGenerator

from matplotlib import pyplot as plt

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
        # plt.savefig(f"./images/{i:04d}.png")


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
    num_runs = 100
    sequence_length = 2  # @param {type:"integer"}
    initial_collect_steps = 128
    collect_steps_per_iteration = 128  # @param {type:"integer"}
    sample_batch_size = 32
    replay_buffer_capacity = 1280  # @param {type:"integer"}

    actor_fc_layers = (64, )
    critic_fc_layers = (64, )

    lr_actor = 1e-4  # @param {type:"number"}
    lr_critic = 1e-4
    lr_alpha = 1e-4
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.9  # @param {type:"number"}
    reward_scale_factor = 1  # @param {type:"number"}
    log_interval = 25  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 50  # @param {type:"integer"}

    # 01 THE ENVIRONMENT
    train_env_py = LaserTracker(visualize=False, speed_restrictions=True, angle_bounds=(40, 140), max_angle_step=20, target_path="circle")
    eval_env_py = LaserTracker(visualize=False, speed_restrictions=True, angle_bounds=(40, 140), target_path="circle")

    # convert to Tensorflow compatible environments
    train_env = tf_py_environment.TFPyEnvironment(train_env_py)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)

#    utils.validate_py_environment(train_env, episodes=5)

    # print(train_env.reset())

    # print(isinstance(train_env, tf_environment.TFEnvironment))
    # print("TimeStep Specs:", train_env.time_step_spec())
    # print("Action Specs:", train_env.action_spec())

    # 01b THE STRATEGY
    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    # 02 THE AGENT
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),   # BoundedArraySpec(shape=(4, ), dtype=dtype('float32'), name='observation', minimum=[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], maximum=[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38])
            train_env.action_spec(),        # BoundedArraySpec(shape=(2, ), dtype=dtype('float32'), name='action', minimum=0, maximum=180)
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )

        # critic_net = value_network.ValueNetwork(
        #     train_env.observation_spec(),
        #     fc_layer_params=critic_fc_layers,
        # )

        critic_net = critic_network.CriticNetwork(
            (train_env.observation_spec(), train_env.action_spec()),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_fc_layers,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform'
        )

        # create optimizer
        actor_opt = tf.optimizers.Adam(learning_rate=lr_actor)
        critic_opt = tf.optimizers.Adam(learning_rate=lr_critic)
        alpha_opt = tf.optimizers.Adam(learning_rate=lr_alpha)

    # tf_agent = reinforce_agent.ReinforceAgent(
    #     train_env.time_step_spec(),  # what is this? (oh it's just state of given time step)
    #     train_env.action_spec(),
    #     actor_network=actor_net,
    #     optimizer=optimizer,
    #     normalize_returns=True,
    #     train_step_counter=train_step_counter
    # )

    with strategy.scope():
        train_step_counter = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            alpha_optimizer=alpha_opt,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step_counter,
        )

        # DONE: We need sequential replay buffer to get sequence of 2 subsequent time steps (for DDPG Agent)

        tf_agent.initialize()  # initialize RL Reinforce agent

    # 03 P0LICY
    eval_policy = tf_agent.policy  # used for evaluation/deployment/production
    collect_policy = tf_agent.collect_policy  # used for data collection
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    # 04 METRICS (see function compute_avg_return @top-of-file)
    # 05 REPLAY BUFFER
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(tf_agent.collect_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=replay_buffer_capacity,
                                                                   )

    # Add an observer that adds to the replay buffer:
    replay_observer = [replay_buffer.add_batch]

    # Collect initial data with random policy
    initial_driver = dynamic_step_driver.DynamicStepDriver(train_env,
                                                           random_policy,
                                                           observers=replay_observer,
                                                           num_steps=initial_collect_steps)
    final_time_step, _ = initial_driver.run()

    driver = dynamic_step_driver.DynamicStepDriver(train_env,
                                                   tf_agent.collect_policy,
                                                   observers=replay_observer,
                                                   num_steps=collect_steps_per_iteration)
    # DONE: This is never gonna end, because we have to define WHEN is the END of the EPISODE

    loss_mean = metrics.Mean()
    for r in range(num_runs):

        final_time_step, policy_state = driver.run()  # DONE: there is a problem with shapes!

        # Read the replay buffer as a Dataset,
        # read batches of 'sample_batch_size' elements, each with 'sequence_length' timesteps:
        dataset = replay_buffer.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=sequence_length)

        loss_mean.reset_state()
        iterator = iter(dataset)
        for _ in range(collect_steps_per_iteration//sample_batch_size):
            t1, probs1 = next(iterator)
            # print(f"obs: {t1.observation.numpy()}, reward: {t1.reward[0].numpy()}")
            # DONE: map two subsequent steps in Trajectory to shape (bs, 2, ... rest, ...)
            loss = tf_agent.train(experience=t1)
            loss_mean.update_state(loss.loss)
            # break
        # action = tf_agent.policy.action(final_time_step)
        print(f"run {r:03d}:  {loss_mean.result():.2f}")
        # break
        # TODO: Make production function to evaluate visually how the agent is doing
        # TODO: DONT FORGET TO DENORMALIZE OBSERVATIONS DURING PRODUCTION with visualization


    # EVALUATION:
    pth_gen = PathGenerator()
    wall = Wall(blit=True)
    trans = Transformations()

    time_step = eval_env.reset()
    while True:

        pred_angles = eval_policy.action(time_step)
        time_step = eval_env.step(pred_angles)

        obs = trans.denormalize_obs(tf.squeeze(time_step.observation).numpy())

        grn_pos = obs[0:2]
        red_pos = obs[2:]

        wall.update(red_pos, grn_pos)
        time.sleep(0.01)
        # break





# DONE: REINFORCE requires full episodes to compute losses.
# TODO: It doesn't care about rewards at all
# TODO: Cummulative Reward for multiple steps?
# DONE: How not to instantly explode into 0,180 angle bounds?
# TODO: make reward function based on distance + efficienty of movement + punishment for not moving
# TODO: enforce boundaries for agent to not go out of bounds of the working area (maybe also implement bigger punishment for doing so)
# TODO: make parralelizable for multiagent training
# TODO: make framework for input type switch (either raw image data or laser positions)
# TODO: implement training with regards to speed_restrictions=True
# TODO: batch sizes