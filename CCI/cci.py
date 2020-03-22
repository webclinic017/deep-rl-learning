import argparse
import sys

import numpy as np
import logging
from tqdm import tqdm
import tensorflow as tf
from keras.utils import to_categorical
from ddqn import DDQN
from env import TradingEnv
from utils.networks import tfSummary

logging.basicConfig(filename='log/cci.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading(DDQN):

    def learning_start(self, env, summary_writer):
        for e in range(args.nb_episodes):
            tqdm_e = tqdm(range(120, 1000), desc='Score', leave=True, unit=" episode")
            time, cumul_reward, done = 0, 0, False
            old_state1, old_state2 = env.reset()
            # print(old_state1.shape, old_state2.shape)
            for x in tqdm_e:
                valid_actions = env.get_valid_actions()
                a = self.random_actions(old_state1, old_state2, valid_actions)
                new_state1, new_state2, r, done, info = env.act(a)
                # print(new_state1.shape, new_state2.shape)
                self.memorize(old_state1, old_state2, a, r, done, new_state1, new_state2)
                old_state1 = new_state1
                old_state2 = new_state2
                cumul_reward += r

                if self.buffer.size() > self.args.batch_size:
                    self.train_agent(batch_size=self.args.batch_size)
                    self.agent.transfer_weights()

                tqdm_e.set_description(
                    "Profit: {}, Cumul reward: {} Step: {}".format(
                        round(env.budget, 2),
                        round(cumul_reward, 2),
                        x
                    )
                )
                tqdm_e.refresh()

            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()


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
    parser.add_argument('--nb_episodes', type=int, default=500, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=5, help="Number of consecutive frames (action repeat)")
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

    trading_env = TradingEnv()
    state_dim = (40,)
    action_dim = 3

    trading_bot = AutoTrading(action_dim, state_dim, args)
    summary_writer = tf.summary.FileWriter("tensorboard/" + args.env)
    trading_bot.learning_start(trading_env, summary_writer)

    trading_bot.save_weights('models/new_nodel')
