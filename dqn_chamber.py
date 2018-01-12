# coding:utf-8

import os
import sys
# import gym
import random
import numpy as np
import cv2, json
from chamber_tracer import ChamberTracer, get_img_and_loc, make_state, get_coord_from_img, parse_coord_str, \
    NUM_ACTIONS, LOG_FOLDER, JSON_DIR, IMG_DIR, FILE_BASE, FRAME_HEIGHT, FRAME_WIDTH, RAW_HEIGHT, RAW_WIDTH, \
    DO_CENTRALIZE, CHEAT_PEEK, ADD_LAST, REWARD_ONLY_NEXT_TRUE, POPULATE_TRUTH, RANDOM_DATA, TEST_FOLDER, \
    NO_EDGE, EDGE_CHECK, EDGE_PENALTY, EPISODE_LIMIT, INTENSITY_MULTIPLIER

import tensorflow as tf
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

##
ENV_NAME = 'ChamberTracer'  # Environment name

STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
NUM_EPISODES = 12000  # Number of episodes the agent plays
# GAMMA = 0.99  # Discount factor
GAMMA = 1  # Discount factor - not sure if discounting makes sense here
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
# INITIAL_REPLAY_SIZE = 10000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
# NUM_REPLAY_MEMORY = 200000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
# SAVE_INTERVAL = 300000  # The frequency with which the network is saved
SAVE_INTERVAL = 50000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode

LOAD_NETWORK = False
TRAIN = True
# LOAD_NETWORK = True
# TRAIN = False
TRAIN_RANDOM = False
if TRAIN_RANDOM:   # train on random img per episode
    # SAVE_NETWORK_PATH = 'saved_networks/{}_h{}_w{}_l{}_p{}_g{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
    #                                                                        POPULATE_TRUTH, RANDOM_DATA)
    # SAVE_SUMMARY_PATH = 'summary/{}_h{}_w{}_l{}_p{}_g{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
    #                                                             POPULATE_TRUTH, RANDOM_DATA)
    SAVE_NETWORK_PATH = 'saved_networks/{}_h{}_w{}_ec{}_ep{}_el{}_im{}_g{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH,
                                                                               EDGE_CHECK, EDGE_PENALTY, EPISODE_LIMIT,
                                                                               INTENSITY_MULTIPLIER, TRAIN_RANDOM)
    SAVE_SUMMARY_PATH = 'summary/{}_h{}_w{}_ec{}_ep{}_el{}_im{}_g{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH,
                                                                            EDGE_CHECK, EPISODE_LIMIT, EDGE_PENALTY,
                                                                            INTENSITY_MULTIPLIER, TRAIN_RANDOM)
elif POPULATE_TRUTH:  # truth gets flushed out too quickly
    SAVE_NETWORK_PATH = 'saved_networks/{}_h{}_w{}_l{}_c{}_r{}_p{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
                                                                           POPULATE_TRUTH)
    SAVE_SUMMARY_PATH = 'summary/{}_h{}_w{}_l{}_c{}_r{}_p{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
                                                                    CHEAT_PEEK, REWARD_ONLY_NEXT_TRUE, POPULATE_TRUTH)
else:   # TRAIN or test random on no truth and old 1-image weights
    ADJUST_LOSS = 0
    REWARD_STATIC_PENALTY = 0
    # SAVE_NETWORK_PATH = 'saved_networks/{}_h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH,
    #                                                                                 NUM_ACTIONS, ADD_LAST, CHEAT_PEEK,
    #                                                                                 ADJUST_LOSS, REWARD_ONLY_NEXT_TRUE,
    #                                                                                 REWARD_STATIC_PENALTY)
    # SAVE_SUMMARY_PATH = 'summary/{}_h{}_w{}_n{}_l{}_c{}_a{}_r{}_s{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, NUM_ACTIONS,
    #                                                                         ADD_LAST, CHEAT_PEEK, ADJUST_LOSS,
    #                                                                         REWARD_ONLY_NEXT_TRUE, REWARD_STATIC_PENALTY)
    SAVE_NETWORK_PATH = 'saved_networks/{}_h{}_w{}_ec{}_ep{}_el{}_im{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH,
                                                                               EDGE_CHECK, EDGE_PENALTY, EPISODE_LIMIT,
                                                                               INTENSITY_MULTIPLIER)
    SAVE_SUMMARY_PATH = 'summary/{}_h{}_w{}_ec{}_ep{}_el{}_im{}'.format(ENV_NAME, FRAME_HEIGHT, FRAME_WIDTH, EDGE_CHECK,
                                                                        EPISODE_LIMIT, EDGE_PENALTY, INTENSITY_MULTIPLIER)

NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        #     self.s, self.q_values, q_network = self.build_network()
        self.s, self.q_values, q_network = self.build_network2()    # input, output, model
        q_network_weights = q_network.trainable_weights

        # Create target network
        # self.st, self.target_q_values, target_network = self.build_network()
        self.st, self.target_q_values, target_network = self.build_network2()   # input, output, model
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', init="he_normal",
                                input_shape=(FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH)))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', init="he_normal"))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', init="he_normal"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH])
        q_values = model(s)

        return s, q_values, model

    def build_network2(self):
        inputs = Input((FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH))

        # Block 1
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4 - more blocks for sizing
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(self.num_actions, name='predictions')(x)

        # Create model.
        model = Model(inputs, x, name='vgg_simple')
        model.summary()     # can also inspect model layers for weights and input/output shapes
        # s = tf.placeholder(tf.float32, [None, FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH])
        q_values = model(inputs)
        return inputs, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])    # action
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)   # should be 9
        # q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)    # old tf
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    # NB this is fine - just take entire image and repeat 4 times if using whole image
    def get_initial_state(self, observation, coord):
        processed_observation = get_img_and_loc(observation, coord)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)

    # FIXME - how should this work?
    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                # FIXME - only save if avg_total_loss decreased
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


###
def main():
    env = ChamberTracer()
    agent = Agent(num_actions=NUM_ACTIONS)    # 8 directions

    if TRAIN:  # Train mode
        if POPULATE_TRUTH:
            env.populate_replay_memory(agent)

        for episode_num in range(NUM_EPISODES):
            terminal = False
            observation = env.reset()    # (single) observation includes location info
            state = env.state   # compartmentalize make_state (initial)

            while not terminal:
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                state = agent.run(state, action, reward, terminal, observation)

            # output action, reward and coords for episode
            num_actions = len(env.actions)
            combined_data = np.asarray(env.actions).reshape((num_actions, 1))
            combined_data = np.append(combined_data, np.asarray(env.rewards).reshape(num_actions, 1), axis=1)
            visited_coords_real = [parse_coord_str(x) for x in env.visited_coords]
            combined_data = np.append(combined_data, np.asarray(visited_coords_real), axis=1)
            # file_name = '{}/h{}_w{}_l{}_p{}_e{}_i{}.txt'.format(LOG_FOLDER, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
            #                                                     POPULATE_TRUTH, episode_num, env.idx)
            file_name = '{}/h{}_w{}_ec{}_ep{}_el{}_im{}_e{}_i{}.txt'.format(LOG_FOLDER, FRAME_HEIGHT, FRAME_WIDTH,
                                                                            EDGE_CHECK, EDGE_PENALTY, EPISODE_LIMIT,
                                                                            INTENSITY_MULTIPLIER, episode_num, env.idx)
            np.savetxt(file_name, combined_data, delimiter=",")
    else:  # Test mode
        total_reward = 0
        for episode_num in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()     # initialise
            state = env.state  # compartmentalize make_state (initial)

            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, reward, terminal, _ = env.step(action)

                total_reward += reward
                # if agent.t % 500==0 or terminal:
                num_steps = len(env.visited_coords)
                if num_steps % 500==0 or terminal:
                    # save_path = '{}/h{}_w{}_l{}_p{}_e{}_i{}_t{}.png'.format(TEST_FOLDER, FRAME_HEIGHT, FRAME_WIDTH,
                    #                                                         ADD_LAST, POPULATE_TRUTH, episode_num,
                    #                                                         env.idx, num_steps)
                    save_path = '{}/h{}_w{}_ec{}_ep{}_el{}_im{}_e{}_i{}_t{}.png'.format(TEST_FOLDER, FRAME_HEIGHT,
                                                                                        FRAME_WIDTH, EDGE_CHECK,
                                                                                        EDGE_PENALTY, EPISODE_LIMIT,
                                                                                        INTENSITY_MULTIPLIER,
                                                                                        episode_num, env.idx, num_steps)
                    # import matplotlib.pyplot as plt
                    # plt.switch_backend('TkAgg')
                    # env.render()
                    env.render(save_path=save_path)
                    # input("Press Enter to continue...")
                    # print('t={}'.format(agent.t), '#visited={}'.format(len(env.visited_coords)),
                    #       'action={}'.format(action), 'reward={}'.format(total_reward), 'terminal={}'.format(terminal))

                    # log episode actions and coords
                    num_actions = len(env.actions)
                    combined_data = np.asarray(env.actions).reshape((num_actions, 1))
                    combined_data = np.append(combined_data, np.asarray(env.rewards).reshape(num_actions, 1), axis=1)
                    visited_coords_real = [parse_coord_str(x) for x in env.visited_coords]
                    combined_data = np.append(combined_data, np.asarray(visited_coords_real), axis=1)
                    file_name = '{}/h{}_w{}_l{}_p{}_e{}_i{}.txt'.format(TEST_FOLDER, FRAME_HEIGHT, FRAME_WIDTH, ADD_LAST,
                                                                        POPULATE_TRUTH, episode_num, env.idx)
                    np.savetxt(file_name, combined_data, delimiter=",")
                elif num_steps > 50000:
                    break   # start new game

                state = np.append(state[:, :, 1:], observation.reshape(FRAME_HEIGHT, FRAME_WIDTH, 1), axis=2)
    1


if __name__ == '__main__':
    # test adding human trace to replay memory
    main()
