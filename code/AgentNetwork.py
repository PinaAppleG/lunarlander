"""
This class wraps a network that works on game pixels of size (height, width)
The network is aimed to be used with the LunarLander environment of OpenAI GYM
It takes in input:
    - nb_frames frames of size (height, width)
It outputs (code may need to be adapted depending on the case):
    - Discrete case: Values of Q for all possible actions
    - Continuous case: two actions in [-1..1]
The agent has two networks:
    - The actual network that approximates Q
    - a target network whose weight updates are much slower than the actual network for stability reasons
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Convolution2D, Input, merge, Lambda, Activation
from keras.optimizers import SGD, Adam

HIDDEN_SIZE = 32


class AgentNetwork:
    def __int__(self, height, width, nb_frames, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        self.model, self.state = self.create_network(height, width, nb_frames, action_size)
        self.target_model, self.target_state = self.create_network(height, width, nb_frames, action_size)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def create_network(self, height, width, nb_frames, action_size):
        print('Building the network')

        S = Input(shape=[nb_frames, height, width])
        c1 = Convolution2D(8, 8, 8, activation='relu')(S)
        c2 = Convolution2D(16, 3, 3, activation='relu')(c1)
        f1 = Flatten()(c2)
        h1 = Dense(HIDDEN_SIZE, activation='relu')(f1)
        V = Dense(action_size)(h1)

        model = Model(input=S, output=V)
        sgd = SGD(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=sgd)
        model.summary()

        return model, S
