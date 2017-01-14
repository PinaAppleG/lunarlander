import numpy as np
import matplotlib.pyplot as plt
import gym
import seaborn

from keras.initializations import normal, identity
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Convolution2D
from keras.optimizers import Adam, SGD

from rl.agents.ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import random

# Loading the environment
ENV_NAME = 'LunarLanderContinuous-v2'

env = gym.make(ENV_NAME)

HIDDEN_SIZE = 256
LEARNING_RATE = 0.001

# Actor model
print("Building the actor")
S = Input(shape=(1,)+env.observation_space.shape)
h0 = Dense(HIDDEN_SIZE / 2, activation='relu')(S)
h1 = Dense(HIDDEN_SIZE, activation='relu')(h0)
Main_engine = Dense(1, activation='tanh')(h1)
Secondary_engine = Dense(1, activation='tanh')(h1)
V = merge([Main_engine, Secondary_engine], mode='concat')
actorModel = Model(input=S, output=V)
# actorModel.summary()
print("Building actor finished")

# Critic model
print("Building the critic")
S = Input(shape=(1,)+env.observation_space.shape)
A = Input(shape=(1,)+env.action_space.shape, name='action2')
w1 = Dense(HIDDEN_SIZE / 2, activation='relu')(S)
a1 = Dense(HIDDEN_SIZE, activation='linear')(A)
h1 = Dense(HIDDEN_SIZE, activation='linear')(w1)
h2 = merge([h1, a1], mode='sum')
h3 = Dense(HIDDEN_SIZE, activation='relu')(h2)
V = Dense(env.action_space.shape[0], activation='linear')(h3)
criticModel = Model(input=[S, A], output=V)
print("Building critic finished")

memory = SequentialMemory(limit=500000, window_length=1)


class LivePlotCallback(Callback):
    def __init__(self, nb_episodes=4000, avgwindow=20):
        self.rewards = np.zeros(nb_episodes) - 1000.0
        self.X = np.arange(1, nb_episodes + 1)
        self.avgrewards = np.zeros(nb_episodes) - 1000.0
        self.avgwindow = avgwindow
        self.rewardbuf = []
        self.episode = 0
        self.nb_episodes = nb_episodes
        plt.ion()
        self.fig = plt.figure()
        self.grphinst = plt.plot(self.X, self.rewards, color='b')[0]
        self.grphavg = plt.plot(self.X, self.avgrewards, color='r')[0]
        plt.ylim([-450.0, 350.0])
        plt.xlabel('Episodes')
        plt.legend([self.grphinst, self.grphavg], ['Episode rewards', '20-episode-average-rewards'])
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='k', linestyle='--')

    def __del__(self):
        self.fig.savefig('monitor/plot.png')

    def on_episode_end(self, episode, logs):
        if self.episode >= self.nb_episodes:
            return
        rw = logs['episode_reward']
        self.rewardbuf.append(rw)
        if len(self.rewardbuf) > self.avgwindow:
            del self.rewardbuf[0]
        self.rewards[self.episode] = rw
        self.avgrewards[self.episode] = np.mean(self.rewardbuf)
        self.plot()
        self.episode += 1

    def plot(self):
        self.grphinst.set_ydata(self.rewards)
        self.grphavg.set_ydata(self.avgrewards)
        plt.draw()
        plt.pause(0.01)


ddpg = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actorModel, critic=criticModel, memory=memory,
                 critic_action_input=criticModel.input[-1], nb_steps_warmup_actor=100, nb_steps_warmup_critic=100)

ddpg.compile(optimizer=[Adam(lr=LEARNING_RATE, decay=2.25e-05), Adam(lr=LEARNING_RATE, decay=2.25e-05)],
             metrics=['mse'])

cbs=[LivePlotCallback(nb_episodes=4000, avgwindow=20)]
ddpg.fit(env, nb_steps=100000, callbacks=cbs)
