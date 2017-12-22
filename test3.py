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
        self.learning_rate = 0.1
        self.action_size = action_size

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(100, input_shape=(8,), activation='sigmoid'))
        self.model.add(Dense(4, activation='sigmoid'))
        self.model.load_weights("./models5/model6.h5")
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
game = WaterWorld(width=256, height=256, num_creeps=15)

# p = PLE(game, reward_values=rewards)
p = PLE(game, fps=30, force_fps=False, display_screen=True,
        reward_values=rewards)

def process_state(current_state):
    processed_state = list()
    processed_state.append(current_state['player_x'])
    processed_state.append(current_state['player_y'])

    processed_state.append(current_state['player_velocity_x'])
    processed_state.append(current_state['player_velocity_y'])

    # LEFT
    min_dist = -9999999
    found = False
    for index in range(len(current_state['creep_pos']['GOOD'])):
        if current_state['player_x'] - 75 <= current_state['creep_pos']['GOOD'][index][0] <= current_state['player_x']:
            if current_state['player_y'] - 10 <= current_state['creep_pos']['GOOD'][index][1] <= current_state['player_y'] + 10:
                if min_dist > current_state['creep_dist']['GOOD'][index]:
                    min_dist = current_state['creep_dist']['GOOD'][index]

    for index in range(len(current_state['creep_pos']['BAD'])):
        if current_state['player_x'] - 75 <= current_state['creep_pos']['BAD'][index][0] <= current_state['player_x']:
            if current_state['player_y'] - 10 <= current_state['creep_pos']['BAD'][index][1] <= current_state['player_y'] + 10:
                if min_dist > current_state['creep_dist']['BAD'][index]:
                    min_dist = current_state['creep_dist']['BAD'][index]
                    found = True
    if found:
        processed_state.append(-min_dist)
    else:
        processed_state.append(min_dist)

    # RIGHT
    min_dist = -9999999
    found = False
    for index in range(len(current_state['creep_pos']['GOOD'])):
        if current_state['player_x'] <= current_state['creep_pos']['GOOD'][index][0] <= current_state['player_x'] + 75:
            if current_state['player_y'] - 10 <= current_state['creep_pos']['GOOD'][index][1] <= current_state['player_y'] + 10:
                if min_dist > current_state['creep_dist']['GOOD'][index]:
                    min_dist = current_state['creep_dist']['GOOD'][index]

    for index in range(len(current_state['creep_pos']['BAD'])):
        if current_state['player_x'] <= current_state['creep_pos']['BAD'][index][0] <= current_state['player_x'] + 75:
            if current_state['player_y'] - 10 <= current_state['creep_pos']['BAD'][index][1] <= current_state['player_y'] + 10:
                if min_dist > current_state['creep_dist']['BAD'][index]:
                    min_dist = current_state['creep_dist']['BAD'][index]
                    found = True
    if found:
        processed_state.append(-min_dist)
    else:
        processed_state.append(min_dist)

    # TOP
    min_dist = -9999999
    found = False
    for index in range(len(current_state['creep_pos']['GOOD'])):
        if current_state['player_y'] - 75 <= current_state['creep_pos']['GOOD'][index][1] <= current_state['player_y']:
            if current_state['player_x'] - 10 <= current_state['creep_pos']['GOOD'][index][0] <= current_state['player_x'] + 10:
                if min_dist > current_state['creep_dist']['GOOD'][index]:
                    min_dist = current_state['creep_dist']['GOOD'][index]

    for index in range(len(current_state['creep_pos']['BAD'])):
        if current_state['player_x'] <= current_state['creep_pos']['BAD'][index][0] <= current_state['player_x'] + 75:
            if current_state['player_y'] - 10 <= current_state['creep_pos']['BAD'][index][1] <= current_state['player_y'] + 10:
                if min_dist > current_state['creep_dist']['BAD'][index]:
                    min_dist = current_state['creep_dist']['BAD'][index]
                    found = True
    if found:
        processed_state.append(-min_dist)
    else:
        processed_state.append(min_dist)

    # BOTTOM
    min_dist = -9999999
    found = False
    for index in range(len(current_state['creep_pos']['GOOD'])):
        if current_state['player_y'] <= current_state['creep_pos']['GOOD'][index][1] <= current_state['player_y'] + 75:
            if current_state['player_x'] - 10 <= current_state['creep_pos']['GOOD'][index][0] <= current_state['player_x'] + 10:
                if min_dist > current_state['creep_dist']['GOOD'][index]:
                    min_dist = current_state['creep_dist']['GOOD'][index]

    for index in range(len(current_state['creep_pos']['BAD'])):
        if current_state['player_y'] <= current_state['creep_pos']['BAD'][index][1] <= current_state['player_y'] + 75:
            if current_state['player_x'] - 10 <= current_state['creep_pos']['BAD'][index][0] <= current_state['player_x'] + 10:
                if min_dist > current_state['creep_dist']['BAD'][index]:
                    min_dist = current_state['creep_dist']['BAD'][index]
                    found = True
    if found:
        processed_state.append(-min_dist)
    else:
        processed_state.append(min_dist)

    return np.array((processed_state,))

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

        current_state = game.getGameState()
        processed_current_state = process_state(current_state)

        action = agent.act(processed_current_state)
        # action = actions[np.random.randint(0, len(actions))]
        reward = p.act(actions[action])
        next_state = game.getGameState()
        game_over = p.game_over()

        print "Current score: " + str(p.score())
    print "Finished with score:" + str(p.score())

