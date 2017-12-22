from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import Dense

from ple import PLE
from ple.games.waterworld import WaterWorld
import random


class Agent:
    def __init__(self, action_size):
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.create_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.2 # exploration rate
        self.learning_rate = 0.01
        self.action_size = action_size

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(200, input_shape=(10000, ), activation='sigmoid'))
        self.model.add(Dense(4, activation='sigmoid'))
        self.model.load_weights("./models4/model16.h5")
        # self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])

rewards = {
    "tick": -0.1,  # each time the game steps forward in time the agent gets -0.1
    "positive": 1,  # each time the agent collects a green circle
    "negative": -5.0,  # each time the agent bumps into a red circle
}

# make a PLE instance.
# use lower fps so we can see whats happening a little easier
game = WaterWorld(width=100, height=100, num_creeps=15)

# p = PLE(game, reward_values=rewards)
p = PLE(game, fps=30, force_fps=False, display_screen=True,
        reward_values=rewards)

p.init()
actions = p.getActionSet()[:-1]
agent = Agent(len(actions))

epochs = 10000000
game_duration = 1000
for epoch in range(epochs):
    p.reset_game()

    for it in range(1000):
        if p.game_over():
            p.reset_game()
            print "Finished with score:" + str(p.score())

        current_state = np.array(p.getScreenGrayscale()).reshape((10000, ))

        action = agent.act(np.array([current_state]))
        # action = actions[np.random.randint(0, len(actions))]
        reward = p.act(actions[action])

        print "Current score: " + str(p.score())
    print "Finished with score:" + str(p.score())

