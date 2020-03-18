import logging
import random
import numpy as np
from keras import Input, Model
from keras.layers import LSTM, Dense, concatenate

from tqdm import tqdm
from agent import Agent

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats

logging.basicConfig(filename='log/ddqn.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.args = args
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.3
        self.buffer_size = 20000
        #
        if len(state_dim) < 3:
            self.tau = 1e-2
        else:
            self.tau = 1.0

        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

    def get_q_valid(self, inp1, inp2, valid_actions):
        q = self.agent.predict(inp1)[0]
        q_valid = [np.nan] * len(q)
        for action in valid_actions:
            q_valid[action] = q[action]
        return q_valid

    # def policy_action(self, s):
    #     """ Apply an espilon-greedy policy to pick next action
    #     """
    #     if random() <= self.epsilon:
    #         return randrange(self.action_dim)
    #     else:
    #         return np.argmax(self.agent.predict(s)[0])

    def random_actions(self, inp1, inp2, valid_actions):
        if np.random.random() > self.epsilon:
            inp1 = np.expand_dims(inp1, axis=0)
            q_valid = self.get_q_valid(inp1, inp2, valid_actions)
            if np.nanmin(q_valid) != np.nanmax(q_valid):
                logging.warning("predict action: {} values: {}".format(np.nanargmax(q_valid), q_valid))
                return np.nanargmax(q_valid)
        action = random.choice(valid_actions)
        # logging.warning("random action: {}".format(action))
        return action

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, s2, a, r, d, new_s1, new_s2, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s1)
        q_targ = self.agent.target_predict(new_s1)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if self.with_per:
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, s2, q_targ)
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, *data):
        """ Store experience in memory buffer
        """

        # if(self.with_per):
        #     q_val = self.agent.predict(state1, state2)
        #     q_val_t = self.agent.target_predict(new_state1, new_state2)
        #     next_best_action = np.argmax(q_val)
        #     new_val = reward + self.gamma * q_val_t[0, next_best_action]
        #     td_error = abs(new_val - q_val)[0]
        # else:
        td_error = 0
        self.buffer.memorize(*data)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        if(self.with_per):
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
