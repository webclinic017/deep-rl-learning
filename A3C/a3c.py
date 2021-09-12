import time
import threading
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense

from A3C.critic import Critic
from A3C.actor import Actor
from A3C.thread import training_thread


class A3C:
    """ Asynchronous Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma=0.85, lr=1e-5):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1
        self.epsilon_min = 0.1
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
        x = Dense(128, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, inp1):
        """ Use the actor to predict the next action to take, using the policy
        """
        inp1 = np.expand_dims(inp1, axis=0)
        p = self.actor.predict(inp1)
        action = np.random.choice(np.arange(self.act_dim), 1, p=p.ravel())[0]
        # logging.warning("a: {}, p: {}".format(action, p))
        return action

    def random_actions(self, inp1):
        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.actor.predict(inp1)
            action = np.argmax(actions_value)
            # logging.warning("action: {} values: {}".format(action, actions_value))
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
        envs = [env] * args.n_threads
        # state_dim = (10, 1)
        # action_dim = 3

        # Create threads
        tqdm_e = tqdm(range(int(args.nb_episodes)), desc='Score', leave=True, unit=" episodes")

        threads = [threading.Thread(
                target=training_thread,
                daemon=True,
                args=(self,
                    args.nb_episodes,
                    envs[i],
                    self.act_dim,
                    args.training_interval,
                    summary_writer,
                    tqdm_e,
                    args.render)) for i in range(args.n_threads)]

        for t in threads:
            t.start()
            time.sleep(0.5)
        try:
            [t.join() for t in threads]
        except KeyboardInterrupt:
            print("Exiting all threads...")
        return None

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
