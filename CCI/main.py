import argparse
import sys
import os
from tqdm import tqdm
import tensorflow as tf
from keras.utils import to_categorical
from A2C.a2c import A2C
from env import TradingEnv
from utils.networks import tfSummary


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
    parser.add_argument('--consecutive_frames', type=int, default=100, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--nb_features', type=int, default=72, help="Number of consecutive frames (action repeat)")
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

    trading_env = TradingEnv(consecutive_frames=args.consecutive_frames, nb_features=args.nb_features)
    state_dim = (args.consecutive_frames, args.nb_features)
    action_dim = 3

    trading_bot = A2C(action_dim, state_dim, args)
    summary_writer = tf.summary.FileWriter("tensorboard/" + args.env)
    trading_bot.learning_start(trading_env, summary_writer)

    trading_bot.save_weights('models/new_nodel')
