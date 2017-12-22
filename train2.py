from collections import deque

import numpy as np
from ple import PLE
from ple.games.waterworld import WaterWorld
from keras.layers import Dense
from keras.models import Sequential
import random
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, action_size):
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999999
        self.create_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.learning_rate = 0.1
        self.action_size = action_size

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(200, input_shape=(10000, ), activation='sigmoid'))
        self.model.add(Dense(4, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        targets = list()
        states = list()
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            targets.append(target_f[0])
            states.append(state[0])
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


rewards = {
    "tick": -0.0,  # each time the game steps forward in time the agent gets -0.1
    "positive": 1,  # each time the agent collects a green circle
    "negative": -1.0,  # each time the agent bumps into a red circle
}

# make a PLE instance.
# use lower fps so we can see whats happening a little easier
game = WaterWorld(width=100, height=100, num_creeps=20)

p = PLE(game, reward_values=rewards)

#
# p = PLE(game, fps=30, force_fps=False, display_screen=True,
#         reward_values=rewards)

p.init()
actions = p.getActionSet()[:-1]
agent = Agent(len(actions))

epochs = 10000000
game_duration = 1000

rewards = []
avg_rewards = []
epsilons = []
steps = []
step = 0
plt.ion()
for epoch in range(epochs):
    p.reset_game()

    for it in range(1000):
        if p.game_over():
            p.reset_game()
            print "Score:" + str(p.score())

        current_state = np.array(p.getScreenGrayscale()).reshape((10000, ))

        action = agent.act(np.array([current_state]))

        reward = p.act(actions[action])
        rewards.append(reward)

        next_state = np.array(p.getScreenGrayscale()).reshape((10000, ))

        game_over = p.game_over()

        agent.remember(np.array([current_state]), action, reward, np.array([next_state]), game_over)
        if len(agent.memory) > 50:
            agent.replay(50)

    steps.append(epoch)
    epsilons.append(agent.epsilon)
    avg_rewards.append(np.average(rewards))
    plt.plot(steps, avg_rewards, 'r')
    plt.plot(steps, epsilons, 'g')

    agent.model.save_weights("./models4/model%d.h5" % epoch, overwrite=False)

    print "Score: " + str(p.score())
    print "Epsilon: " + str(agent.epsilon)
