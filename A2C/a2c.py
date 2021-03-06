import os
import random
import numpy as np
import logging
import joblib
from keras.utils.np_utils import to_categorical

os.makedirs('log', exist_ok=True)
logging.basicConfig(filename='log/a2c.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Dense, Flatten, LSTM

from .critic import Critic
from .actor import Actor
from utils.networks import tfSummary
from utils.stats import gather_stats


class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma=0.99, lr=1e-4):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.epsilon_min = 0.2
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Dense(64, activation='relu')(inp)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, inp1):
        """ Use the actor to predict the next action to take, using the policy
        """
        inp1 = np.expand_dims(inp1, axis=0)
        p = self.actor.predict(inp1)
        action = np.random.choice(np.arange(self.act_dim), 1, p=p.ravel())[0]
        logging.warning("a: {}, p: {}".format(action, p))
        return action

    def random_actions(self, inp1):
        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.actor.predict(inp1)
            action = np.argmax(actions_value)
            logging.warning("action: {} values: {}".format(action, actions_value))
        else:
            action = np.random.randint(0, self.act_dim)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.999
        return action

    def discount(self, r, done, a):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        discredit = 0
        for t in reversed(range(0, len(r))):
            reward = r[t]
            cumul_r = reward + (cumul_r * self.gamma) + discredit
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, actions)
        state_values = self.critic.predict(np.array(states))

        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization

        self.a_opt([states, actions, advantages])
        self.c_opt([states, discounted_rewards])

    def train(self, env, args, summary_writer):
        """ Main A2C Training Algorithm
        """

        results = []

        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []

            while not done:
                # if args.render: env.render()
                # Actor picks an action (following the policy)
                valid_actions = env.get_valid_actions()
                action = self.random_actions(old_state)
                # if action not in valid_actions:
                #     action = 0  # do not do any think

                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, info = env.step(action)

                time += 1
                # logging.warning(info)
                # Memorize (s, a, r) for training
                if not done:
                    actions.append(to_categorical(action, self.act_dim))
                    rewards.append(r)
                    states.append(old_state)
                    # Update current state
                    old_state = new_state
                    cumul_reward += r

            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)

            # Gather stats every episode for plotting
            # if(args.gather_stats):
            #     mean, stdev = gather_stats(self, env)
            #     results.append([e, mean, stdev])
            tqdm_e.set_description(f"total loss: {round(info['max_loss'], 1)} total profit {round(info['max_profit'], 1)} real profit {round(info['profit'], 1)}")
            tqdm_e.refresh()
            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            budget = tfSummary('profit', info['profit'])
            summary_writer.add_summary(score, global_step=e)
            summary_writer.add_summary(budget, global_step=e)
            summary_writer.flush()

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
