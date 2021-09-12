""" Training thread for A3C
"""

import numpy as np
from threading import Thread, Lock
from keras.utils import to_categorical
from utils.networks import tfSummary
import logging

logging.basicConfig(filename='log/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

episode = 0
lock = Lock()


def training_thread(agent, Nmax, env, action_dim, f, summary_writer, tqdm, render):
    """ Main A2C Training Algorithm
    """

    global episode
    while episode < Nmax:

        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        actions, states, rewards = [], [], []

        while not done:
            # if args.render: env.render()
            # Actor picks an action (following the policy)
            valid_actions = env.get_valid_actions()
            action = agent.policy_action(old_state)
            if action not in valid_actions:
                action = 0  # do not do any think
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, info = env.step(action)
            # logging.warning(info)
            # Memorize (s, a, r) for training
            actions.append(to_categorical(action, action_dim))
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1
            # Display score
            # if done:
                # done = True if info['total_profit'] > 1000 else False
                # Train using discounted rewards ie. compute updates
        lock.acquire()
        agent.train_models(states, actions, rewards, done)
        lock.release()
        # actions, states, rewards = [], [], []

        # Gather stats every episode for plotting
        # if(args.gather_stats):
        #     mean, stdev = gather_stats(self, env)
        #     results.append([e, mean, stdev])
        with lock:
            tqdm.set_description(
                f"total loss: {round(info['max_loss'], 1)} total profit {round(info['max_profit'], 1)} real profit {round(info['profit'], 1)}")
            tqdm.update(1)
            if episode < Nmax:
                episode += 1

        # # Export results for Tensorboard
        # score = tfSummary('score', cumul_reward)
        budget = tfSummary('profit', info['profit'])
        # summary_writer.add_summary(score, global_step=e)
        summary_writer.add_summary(budget, global_step=episode)
        summary_writer.flush()
