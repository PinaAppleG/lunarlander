import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
import json

from _util import preprocess_env, phi
from memory import ReplayMemory
from AgentNetwork import AgentNetwork


def play(train_indicator=0):  # 1=train, 2=run simply
    MEMORY_SIZE = 100  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    BATCH_SIZE = 16  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    GAMMA = 0.9
    TAU = 0.001
    LEARNING_RATE = 0.00001

    nb_actions = 4
    nb_frames = 4
    height = 40
    width = 60

    nb_episodes = 10  # CHANGE TO BIGGER VALUE BEFORE RUNNING
    max_steps = 200  # CHANGE TO BIGGER VALUE BEFORE RUNNING

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

            replay_memory.append([phi(temp), a_t, r_t, phi(s_t)], env.game_over)
            cum_reward += GAMMA ** t * r_t

            batch, batch_state = replay_memory.mini_batch(size=agent.BATCH_SIZE)
            
            x_batch = []
            y_batch = []
            for j,transition in enumerate(batch):
             
                phi_j = np.array(batch[j][0])
                # Converting to correct size for Keras
                phi_j = phi_j[None,:,:,:]
                # Output of size 4
                q_t = agent.model.predict(phi_j)[0]
                

                if batch_state[j] == True:
                    # Terminal j+1
                    y_j = [transition[2]]*nb_actions
                else:
                    # Non-terminal j+1
                    phi_j_plus_1 = np.array(batch[j][3])
                    phi_j_plus_1 = phi_j_plus_1[None,:,:,:]
                    q_t_plus_1 = agent.target_model.predict(phi_j_plus_1)[0]
                    
                    max_idx = np.argmax(q_t_plus_1)
                    max_val = q_t_plus_1[max_idx]
                   
                    # rj + gamma * max a0 Q(j+1; a0; theta)
                    y_j = q_t 
                    y_j[max_idx] = transition[2]+GAMMA*max_val
                    
                x_batch.append(phi_j)
                y_batch.append(np.array(y_j)[None,:])

            #Perform a gradient descent step on (yj - Q(j ; aj ; theta))2 according to equation 3
            callback = agent.model.train_on_batch(np.array(x_batch)[0],np.array(y_batch)[0])
            print('Loss on batch: '+str(loss))
            agent.target_train()
            loss_v.append(callback)

            if env.game_over:
                break
        print(cum_reward)
        reward_v.append(cum_reward)

        if (train_indicator):
            loss_v.append(loss)
            print("Now we save model")
            agent.model.save_weights("model.h5f", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(agent.model.to_json(), outfile)
    env.close()
    print("finish")


if __name__ == "__main__":
    play(1)
