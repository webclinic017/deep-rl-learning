import argparse
import sys
from ddqn import DDQN
from env import TradingEnv


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
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=10, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--nb_features', type=int, default=5, help="Number of consecutive frames (action repeat)")
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

    trading_env = TradingEnv(consecutive_frames=args.consecutive_frames,
                             nb_features=args.nb_features, dataset='../data/test_5m.csv', strategy='train')
    test_env = TradingEnv(consecutive_frames=args.consecutive_frames,
                          nb_features=args.nb_features, dataset='../data/test_5m.csv', strategy='test')

    state_dim = args.nb_features * args.consecutive_frames + 2
    action_dim = 3

    trading_bot = DDQN(action_dim, state_dim, args)
    trading_bot.train(trading_env, args, test_env)

    trading_bot.save_weights('models/new_nodel')
