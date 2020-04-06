"""This is a simple implementation of [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)"""
import argparse
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='log/random.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
from env import TradingEnv


class CuriosityNet:
    def __init__(
            self,
            n_a,
            n_s,
            lr=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.9999,
            min_epsilon=0.01,
            replace_target_iter=300,
            memory_size=10000,
            batch_size=128,
            output_graph=False,
    ):
        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.s_encode_size = 1000       # give a hard job for predictor to learn

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_s * 2 + 2))
        self.tfs, self.tfa, self.tfr, self.tfs_, self.pred_train, self.dqn_train, self.q = \
            self._build_nets()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_nets(self):
        tfs = tf.placeholder(tf.float32, [None, self.n_s], name="s")    # input State
        tfa = tf.placeholder(tf.int32, [None, ], name="a")              # input Action
        tfr = tf.placeholder(tf.float32, [None, ], name="ext_r")        # extrinsic reward
        tfs_ = tf.placeholder(tf.float32, [None, self.n_s], name="s_")  # input Next State

        # fixed random net
        with tf.variable_scope("random_net"):
            rand_encode_s_ = tf.layers.dense(tfs_, self.s_encode_size)

        # predictor
        ri, pred_train = self._build_predictor(tfs_, rand_encode_s_)

        # normal RL model
        q, dqn_loss, dqn_train = self._build_dqn(tfs, tfa, ri, tfr, tfs_)
        return tfs, tfa, tfr, tfs_, pred_train, dqn_train, q

    def _build_predictor(self, s_, rand_encode_s_):
        with tf.variable_scope("predictor"):
            net = tf.layers.dense(s_, 128, tf.nn.relu)
            out = tf.layers.dense(net, self.s_encode_size)

        with tf.name_scope("int_r"):
            ri = tf.reduce_sum(tf.square(rand_encode_s_ - out), axis=1)  # intrinsic reward
        train_op = tf.train.RMSPropOptimizer(self.lr, name="predictor_opt").minimize(
            tf.reduce_mean(ri), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "predictor"))

        return ri, train_op

    def _build_dqn(self, s, a, ri, re, s_):
        with tf.variable_scope('eval_net'):
            e0 = tf.layers.dense(s, 512, tf.nn.relu)
            e1 = tf.layers.dense(e0, 128, tf.nn.relu)
            q = tf.layers.dense(e1, self.n_a, name="q")
        with tf.variable_scope('target_net'):
            t0 = tf.layers.dense(s_, 512, tf.nn.relu)
            t1 = tf.layers.dense(t0, 128, tf.nn.relu)
            q_ = tf.layers.dense(t1, self.n_a, name="q_")

        with tf.variable_scope('q_target'):
            q_target = re + ri + self.gamma * tf.reduce_max(q_, axis=1, name="Qmax_s_")

        with tf.variable_scope('q_wrt_a'):
            a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
            q_wrt_a = tf.gather_nd(params=q, indices=a_indices)

        loss = tf.losses.mean_squared_error(labels=q_target, predictions=q_wrt_a)   # TD error
        train_op = tf.train.RMSPropOptimizer(self.lr, name="dqn_opt").minimize(
            loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eval_net"))
        return q, loss, train_op

    def store_transition(self, s, a, r, s_):            
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    @staticmethod
    def get_q_valid(q, valid_action):
        q_valid = [np.nan] * len(q)
        for action in valid_action:
            q_valid[action] = q[action]
        return q_valid

    # def choose_action(self, observation, valid_action):
    #     # to have batch dimension when feed into tf placeholder
    #     s = observation[np.newaxis, :]
    #
    #     if np.random.uniform() > self.epsilon:
    #         # forward feed the observation and get q value for every actions
    #         actions_value = self.sess.run(self.q, feed_dict={self.tfs: s})[0]
    #         q_valid = self.get_q_valid(actions_value, valid_action)
    #         if np.nanmin(q_valid) != np.nanmax(q_valid):
    #             action = np.nanargmax(q_valid)
    #             logging.warning("action: {} | actions_value: {} | epsilon: {}".format(action, q_valid, round(self.epsilon, 2)))
    #             return action
    #
    #     action = np.random.randint(0, self.n_a)
    #     return action

    def choose_action(self, observation, valid_action):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q, feed_dict={self.tfs: s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_a)
        return action

    def action(self, observation, valid_action):
        # to have batch dimension when feed into tf placeholder
        s = observation[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q, feed_dict={self.tfs: s})[0]
        action = np.argmax(actions_value)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(top, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        bs, ba, br, bs_ = batch_memory[:, :self.n_s], batch_memory[:, self.n_s], \
            batch_memory[:, self.n_s + 1], batch_memory[:, -self.n_s:]
        self.sess.run(self.dqn_train, feed_dict={self.tfs: bs, self.tfa: ba, self.tfr: br, self.tfs_: bs_})
        if self.learn_step_counter % 100 == 0:     # delay training in order to stay curious
            self.sess.run(self.pred_train, feed_dict={self.tfs_: bs_})
        self.learn_step_counter += 1

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=500000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=3, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--nb_features', type=int, default=10, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='Autotrading',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    env = TradingEnv(consecutive_frames=args.consecutive_frames, nb_features=args.nb_features,
                     dataset='../data/train_1h.csv')
    test_env = TradingEnv(consecutive_frames=args.consecutive_frames, nb_features=args.nb_features,
                          dataset='../data/test_1h.csv', strategy='test')
    state_dim = (args.consecutive_frames, args.nb_features)
    action_dim = 4

    dqn = CuriosityNet(n_a=action_dim, n_s=30, lr=0.01, output_graph=False)
    ep_steps = []
    for epi in range(20000):
        s = env.reset()
        steps = 0
        cul_reaward = 0
        while True:
            # env.render()
            valid_actions = env.get_valid_actions()
            a = dqn.choose_action(s, valid_actions)
            s_, r, done, info = env.act(a)
            dqn.store_transition(s, a, r, s_)
            dqn.learn()
            cul_reaward += r

            if done:
                logging.warning("Cumun Reward: {} | Done: {} | Step: {}".format(cul_reaward, done, env.t))
                # print('Train Epi: ', epi, "| steps: ", steps, "| cul_reaward: ", cul_reaward)
                ep_steps.append(steps)
                break

            s = s_
            steps += 1

        s = test_env.reset()
        while True:
            valid_actions = test_env.get_valid_actions()
            a = dqn.action(s, valid_actions)
            info['action'] = a
            # logging.warning("info: {}".format(info))
            s_, r, done, info = test_env.act(a)
            s = s_
            if done:
                print('Test Epi: {} | steps: {} | Budget: {} | Cul Reward: {}'.format(epi, test_env.t, round(info['budget'], 2), cul_reaward))
                break

    plt.plot(ep_steps)
    plt.ylabel("steps")
    plt.xlabel("episode")
    plt.show()
