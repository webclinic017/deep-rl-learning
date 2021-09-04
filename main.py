""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import time

import argparse
import tensorflow as tf

from A2C.a2c import A2C

from keras.backend.tensorflow_backend import set_session

from A2C.env import TradingEnv
from utils.networks import get_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='A2C', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=500000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=14,
                        help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=32, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', default=False, action='store_true',
                        help="Render environment while training")
    parser.add_argument('--env', type=str, default='BTCUSDT', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--test', type=bool, default=False, help='Train of test')
    parser.add_argument('--actor_path', type=str)
    parser.add_argument('--critic_path', type=str)
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())
    os.makedirs(args.type + "/tensorboard_" + args.env, exist_ok=True)
    print("Log dir" + args.type + "/tensorboard_" + args.env)
    for file in os.listdir(args.type + "/tensorboard_" + args.env):
        os.remove(args.type + "/tensorboard_" + args.env + '/' + file)
    summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)

    env = TradingEnv(consecutive_frames=args.consecutive_frames)
    env.reset()
    state_dim = (14,)
    action_dim = 3
    act_range = 2
    algo = A2C(action_dim, state_dim, args.consecutive_frames)

    # Train
    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    # if args.gather_stats:
    #     df = pd.DataFrame(np.array(stats))
    #     df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}_{}'.format(exp_dir,
                                                         args.type,
                                                         args.env,
                                                         args.nb_episodes,
                                                         args.batch_size,
                                                         int(time.time()))

    algo.save_weights(export_path)


if __name__ == "__main__":
    main()
