import logging
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
logging.basicConfig(filename='log/random.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = False
        self.action_dim = action_dim
        self.state_dim = state_dim
        #
        self.lr = 1e-3
        self.gamma = 0.95
        self.alpha = 0.75
        self.delta = 0.95
        self.discount_rate = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.buffer_size = 20000
        #
        # if(len(self.state_dim) < 3):
        self.tau = 1e-2
        # else:
        #     self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, False)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, False)

    def policy_action(self, s1, s2):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            s1 = np.expand_dims(s1, axis=0)
            s2 = np.expand_dims(s2, axis=0)
            policy_value = self.agent.predict(s1, s2)[0]
            logging.warning(policy_value)
            return np.argmax(policy_value)

    def get_action(self, s1, s2):
        """ Apply an espilon-greedy policy to pick next action
        """
        # logging.warning(self.agent.predict(s))
        s1 = np.expand_dims(s1, axis=0)
        s2 = np.expand_dims(s2, axis=0)
        policy_value = self.agent.predict(s1, s2)[0]
        logging.warning(policy_value)
        return np.argmax(policy_value)

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s1, s2, a, r, d, new_s1, new_s2, idx = self.buffer.sample_batch(batch_size)

        discounted_rewards = self.discount(r)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s1, s2)
        next_q = self.agent.predict(new_s1, new_s2)
        q_targ = self.agent.target_predict(new_s1, new_s2)

        for i in range(s1.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if self.with_per:
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s1, s2, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay

    def test(self, env, args, step):
        """ Main DDQN Training Algorithm
        """
        # Reset episode
        time, cumul_reward, done = 0, 0, False
        s1, s2 = env.reset()

        while True:
            if args.render: env.render()
            # Actor picks an action (following the policy)
            a = self.get_action(s1, s2)
            # Retrieve new state, reward, and whether the state is terminal
            n_s1, n_s2, r, done, info = env.act(a)
            s1 = n_s1
            s2 = n_s2
            cumul_reward += r
            time += 1
            info['step'] = step
            if info['end_ep'] or done:
                logging.warning(info)
                break

    def train(self, train_env, args, test_env):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in range(args.nb_episodes):
            # Reset episode
            time, cumul_reward, done = 0, 0, False
            s1, s2 = train_env.reset()

            while not done:
                if args.render: train_env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(s1, s2)
                # Retrieve new state, reward, and whether the state is terminal
                new_s1, new_s2, r, done, info = train_env.act(a)
                # logging.warning("actions: {}".format(a))
                print("Actions: {} | Budget: {} | Steps: {} | Episode: {} | Diff: {} | Reward: {}".format(
                    a, round(info['budget'], 2),
                    train_env.t, e, round(info['diff'], 2), round(r, 2))
                )
                # Memorize for experience replay
                self.memorize(s1, s2, a, r, done, new_s1, new_s2)
                # Update current state
                s1 = new_s1
                s2 = new_s2
                cumul_reward += r
                time += 1

                # tqdm_e.set_description("Budget: {} | Steps: {} | Actions: {}".format(round(info['budget'], 3), train_env.t, a))
                # tqdm_e.refresh()
                # print("buffer count: {} batch size {}".format(self.buffer.count, args.batch_size))
                # Train DDQN and transfer weights to target network
                if self.buffer.count > args.batch_size:
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            if e % 50 == 0:
                self.test(test_env, args, e)
            # Display score
            # total_profit = self.test(summary_writer, e)

            # Export results for Tensorboard
            # score = tfSummary('score', cumul_reward)
            # summary_writer.add_summary(score, global_step=e)
            # budget = tfSummary('budget', total_profit)
            # summary_writer.add_summary(budget, global_step=e)
            # summary_writer.flush()

        return results

    def memorize(self, state1, state2, action, reward, done, new_state1, new_state2):
        """ Store experience in memory buffer
        """
        if self.with_per:
            s1 = np.expand_dims(state1, axis=0)
            s2 = np.expand_dims(state2, axis=0)
            n_s1 = np.expand_dims(new_state1, axis=0)
            n_s2 = np.expand_dims(new_state2, axis=0)
            q_val = self.agent.predict(s1, s2)
            q_val_t = self.agent.target_predict(n_s1, n_s2)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state1, state2, action, reward, done, new_state1, new_state2, td_error)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        if self.with_per:
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
