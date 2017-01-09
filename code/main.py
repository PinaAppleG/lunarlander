import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import json

from _util import preprocess_env, phi
from memory import ReplayMemory
from AgentNetwork import AgentNetwork


def play(train_indicator=0):  # 1=train, 2=run simply
    MEMORY_SIZE = 10  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    BATCH_SIZE = 4  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    GAMMA = 0.9
    TAU = 0.001
    LEARNING_RATE = 0.01

    nb_actions = 4
    nb_frames = 4
    height = 80
    width = 120

    nb_episodes = 10  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    max_steps = 100  # CHANGE TO BIGGER VALUE BEFORE RUNNING

    epsilon = 0.01

    replay_memory = ReplayMemory(MEMORY_SIZE)
    agent = AgentNetwork(height, width, nb_frames, nb_actions, BATCH_SIZE, TAU, LEARNING_RATE)

    env = gym.make('LunarLander-v2')

    for episode in range(nb_episodes):
        env.reset()

        loss_v = []
        reward_v = []

        s_t = [preprocess_env(env)]

        for t in range(max_steps):
            loss = 0
            cum_reward = 0

            if np.random.rand() < epsilon:
                a_t = np.random.randint(4)
            else:
                q = agent.model.predict(phi(s_t)[None, :, :, :])[0]
                a_t = np.argmax(q)

            _, r_t, done, _ = env.step(a_t)
            x_t1 = preprocess_env(env)

            temp = s_t
            s_t.append(a_t)
            s_t.append(x_t1)

            replay_memory.appnd([phi(temp), a_t, r_t, phi(s_t)], done)
            cum_reward += GAMMA ** t * r_t

            batch, batch_state = replay_memory.mini_batch(size=BATCH_SIZE)

            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            for j in range(len(batch)):
                if batch_state:
                    y_t[j] = rewards[j]
                else:
                    target_q = agent.target_model.predict(new_states[j][None, :, :, :])[0]
                    max_idx = np.argmax(target_q)

                    y_t[j] = agent.model.predict(states[j][None, :, :, :])[0]
                    y_t[j][max_idx] = rewards[j] + GAMMA * target_q[max_idx]

            if (train_indicator):
                loss += agent.model.train_on_batch(states, y_t)
                agent.target_train()

            if env.game_over:
                break

        reward_v.append(cum_reward)

        if (train_indicator):
            loss_v.append(loss)
            print("Now we save model")
            agent.model.save_weights("model.h5f", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(agent.model.to_json(), outfile)
    env.end()
    print("finish")


if __name__ == "__main":
    play(1)
