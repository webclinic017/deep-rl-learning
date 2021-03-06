""" Load and display pre-trained model in OpenAI Gym Environment
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from A2C.a2c import A2C
from A3C.a3c import A3C
from DDQN.ddqn import DDQN
from DDPG.ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from utils.atari_environment import AtariEnvironment
from utils.environment import Environment
from utils.networks import get_session

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='DDQN', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--consecutive_frames', type=int, default=10,
                        help="Number of consecutive frames (action repeat)")
    #
    parser.add_argument('--model_path', type=str, help="Number of training episodes")
    parser.add_argument('--actor_path', type=str, help="Number of training episodes")
    parser.add_argument('--critic_path', type=str, help="Batch size (experience replay)")
    #
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--test', type=bool, default=False, help='Train of test')
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

    env = Environment(start_step=11, windows=args.consecutive_frames, dataset='train1hour')
    env.reset()
    state_dim = (2,)
    action_dim = 2
    act_range = 2

    # Pick algorithm to train
    if (args.type == "DDQN"):
        algo = DDQN(action_dim, state_dim, args)
        algo.load_weights(args.model_path)
    elif (args.type == "A2C"):
        algo = A2C(action_dim, state_dim, args.consecutive_frames)
        algo.load_weights(args.actor_path, args.critic_path)
    elif (args.type == "A3C"):
        algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
        algo.load_weights(args.actor_path, args.critic_path)
    elif (args.type == "DDPG"):
        algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)
        algo.load_weights(args.actor_path, args.critic_path)

    # Display agent
    old_state, time = env.reset(), 0
    done = False
    while not done:
        a = algo.policy_action(old_state)
        old_state, r, done, info = env.step(a)
        time += 1
        print(info)


if __name__ == "__main__":
    main()
