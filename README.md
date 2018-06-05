# Advanced Deep RL in Keras

Implementation of various Deep Reinforcement Learning algorithms:

- [x] Synchronous N-step Advantage Actor Critic ([A2C](https://github.com/germain-hug/Advanced-Deep-RL-Keras#n-step-advantage-actor-critic-a2c))
- [x] Asynchronous N-step Advantage Actor-Critic ([A3C](https://github.com/germain-hug/Advanced-Deep-RL-Keras#n-step-asynchronous-advantage-actor-critic-a3c))
- [ ] Deep Deterministic Policy Gradient with Parameter Noise ([DDPG](https://github.com/germain-hug/Advanced-Deep-RL-Keras#deep-deterministic-policy-gradient-ddpg))
- [ ] Deep Deterministic Policy Gradient with Hindsight Experience Replay ([DDPG + HER](https://github.com/germain-hug/Advanced-Deep-RL-Keras#deep-deterministic-policy-gradient-with-hindsight-experience-replay-ddpg--her))
- [ ] REINFORCE
- [ ] Deep Q-Learning (DQN)
- [ ] Dueling DQN (DDQN)
- [ ] Rainbow
- [ ] Proximal Policy Optimization (PPO)
- [ ] Impala
- [ ] Spiral

### Prerequisites

This implementation requires keras 2.1.6, as well as OpenAI gym.
```
pip install gym keras==2.1.6
```

## N-step Advantage Actor Critic (A2C)
The Actor-Critic algorithm is a model-free, off-policy method where the critic acts as a value-function approximator, and the actor as a policy-function approximator. When training, the critic predicts the TD-Error and guides the learning of both itself and the actor. In practice, we approximate the TD-Error using the Advantage function. For more stability, we use a shared computational backbone across both networks, as well as an N-step formulation of the discounted rewards. We also incorporate an entropy regularization term ("soft" learning) to encourage exploration. While A2C is simple and efficient, running it on Atari Games quickly becomes intractable due to long computation time. However, one can parallelize computation using multi-threaded agents, which is the point of A3C. We test A2C on the Cartpole-V1 environment.

```bash
python3 A2C/main.py --env CartPole-v1 --nb_episodes 10000 --render
```
<br />
<div align="center">
<img width="40%" src ="https://github.com/germain-hug/Advanced-Deep-RL-Keras/blob/master/A2C/results/a2c.png?raw=true" />
<img width="30%" src ="https://github.com/germain-hug/Advanced-Deep-RL-Keras/blob/master/A2C/results/a2c.gif?raw=true" />
<p style="text-align=center";> A2C Average Score and Results [Cartpole-V1] </p></div>  
<br />

## N-step Asynchronous Advantage Actor Critic (A3C)
In a similar fashion as the A2C algorithm, the implementation of A3C incorporates asynchronous weight updates, allowing for much faster computation. We use multiple agents to perform gradient ascent asynchronously, over multiple threads. We test A3C on the Atari Breakout environment.

```bash
python3 A3C/main.py --env CartPole-v1 --nb_episodes 1000 --n_threads 16
```

## Deep Deterministic Policy Gradient (DDPG)
The DDPG algorithm is a model-free, off-policy algorithm for continuous action spaces. Similarly to A2C, it is an actor-critic algorithm in which the actor is trained on a deterministic target policy, and the critic predicts Q-Values. In order to reduce variance and increase stability, we use experience replay and separate target networks. Moreover, as hinted by [OpenAI](https://blog.openai.com/better-exploration-with-parameter-noise/), we encourage exploration through parameter space noise (as opposed to traditional action space noise). We test DDPG on the Lunar Lander environment.

```bash
python3 DDPG/main.py --nb_episodes 1000 --n_threads 16
```

## Deep Deterministic Policy Gradient with Hindsight Experience Replay (DDPG + HER)
Hindsight Experience Replay (HER) brings an improvement to both discrete and continuous action space off-policy methods. It is particularly suited for robotics application as it enables efficient learning from _sparse_ and _binary_ rewards. HER formulates the problem as a multi-goal task, where new goals are being sampled at the start of each episode through a specific strategy.

# Visualization

# Acknowledgments

- Memory Buffer template by [Patrick Emami](http://pemami4911.github.io/)
- Atari Environment Helper Class template by [ShanHaoYu](https://github.com/ShanHaoYu/Deep-Q-Network-Breakout/blob/master/environment.py)
- Atari Environment Wrappers by [OpenAI](github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)

# References (Papers)

- [Advantage Actor Critic (A2C)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
- [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](http://proceedings.mlr.press/v32/silver14.pdf)
- [Hindsight Experience Replay (HER)](https://arxiv.org/pdf/1707.01495.pdf)
