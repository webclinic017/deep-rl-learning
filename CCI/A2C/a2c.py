import random
import numpy as np
import logging

logging.basicConfig(filename='../log/a2c.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

from tqdm import tqdm
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, concatenate

from A2C.critic import Critic
from A2C.actor import Actor


class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma=0.8, lr=1e-4):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k,) + env_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.999
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers"""
        initial_input = Input(shape=(2, 40))
        secondary_input = Input(shape=(2,))

        lstm = LSTM(128, return_sequences=True)(initial_input)
        lstm = LSTM(128, return_sequences=True)(lstm)
        lstm = LSTM(128)(lstm)
        dense = Dense(128, activation='relu')(secondary_input)
        merge = concatenate([lstm, dense])

        x = Dense(64, activation='relu')(merge)
        out_dense = Dense(64, activation='relu')(x)
        output = Dense(64, activation='relu')(out_dense)
        model = Model(inputs=[initial_input, secondary_input], outputs=output)
        return model

    def policy_action(self, inp1, inp2):
        """ Use the actor to predict the next action to take, using the policy
        """
        p = self.actor.predict(inp1, inp2)
        action = np.random.choice(np.arange(self.act_dim), 1, p=p.ravel())[0]
        logging.warning("a: {}, p: {}".format(action, p))
        return action

    def get_q_valid(self, inp1, inp2, valid_actions):
        q = self.actor.predict(inp1, inp2)[0]
        q_valid = [np.nan] * len(q)
        for action in valid_actions:
            q_valid[action] = q[action]
        return q_valid

    def random_actions(self, inp1, inp2, valid_actions):
        if np.random.random() > self.epsilon:
            inp1 = np.expand_dims(inp1, axis=0)
            inp2 = np.expand_dims(inp2, axis=0)
            q_valid = self.get_q_valid(inp1, inp2, valid_actions)
            if np.nanmin(q_valid) != np.nanmax(q_valid):
                logging.warning("predict action: {} values: {}".format(np.nanargmax(q_valid), q_valid))
                return np.nanargmax(q_valid)
        action = random.sample(valid_actions, 1)[0]
        logging.warning("random action: {}".format(action))
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

    def train_models(self, state1, state2, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards, done, actions)
        s1 = np.array(state1)
        s2 = np.array(state2)
        state_values = self.critic.predict(s1, s2)

        advantages = discounted_rewards - np.reshape(state_values, len(state_values))
        # Networks optimization

        self.a_opt([s1, s2, actions, advantages])
        self.c_opt([s1, s2, discounted_rewards])

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

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
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, info = env.step(a)
                # logging.warning(info)
                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, self.act_dim))
                rewards.append(r)
                states.append(old_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Display score

            # done = True if info['total_profit'] > 1000 else False
            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)

            # Gather stats every episode for plotting
            # if(args.gather_stats):
            #     mean, stdev = gather_stats(self, env)
            #     results.append([e, mean, stdev])
            tqdm_e.set_description("Profit: " + str(info['total_profit']))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
