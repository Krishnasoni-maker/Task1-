pip install pygame gym numpy tensorflow
import gym
from gym import spaces
import numpy as np
import pygame
import random

class ObstacleAvoidanceEnv(gym.Env):
    def __init__(self):
        super(ObstacleAvoidanceEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 actions: left, right, up, down
        self.observation_space = spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)

        self.screen_width = 800
        self.screen_height = 600
        self.robot_size = 20
        self.obstacle_size = 20

        self.obstacle_list = [(random.randint(0, self.screen_width), random.randint(0, self.screen_height)) for _ in range(10)]
        
        self.robot_pos = [self.screen_width // 2, self.screen_height // 2]

    def reset(self):
        self.robot_pos = [self.screen_width // 2, self.screen_height // 2]
        return self._get_state()

    def step(self, action):
        if action == 0:  # left
            self.robot_pos[0] -= 10
        elif action == 1:  # right
            self.robot_pos[0] += 10
        elif action == 2:  # up
            self.robot_pos[1] -= 10
        elif action == 3:  # down
            self.robot_pos[1] += 10

        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.screen_width)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.screen_height)

        reward = 1
        done = False

        for obstacle in self.obstacle_list:
            if self._is_collision(self.robot_pos, obstacle):
                reward = -10
                done = True
                break

        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        screen.fill((255, 255, 255))

        pygame.draw.rect(screen, (0, 0, 255), (*self.robot_pos, self.robot_size, self.robot_size))

        for obstacle in self.obstacle_list:
            pygame.draw.rect(screen, (255, 0, 0), (*obstacle, self.obstacle_size, self.obstacle_size))

        pygame.display.flip()

    def _get_state(self):
        state = np.zeros((10, 10, 3), dtype=np.uint8)
        for obstacle in self.obstacle_list:
            x, y = obstacle
            state[x//80, y//60] = [255, 0, 0]

        x, y = self.robot_pos
        state[x//80, y//60] = [0, 0, 255]
        return state

    def _is_collision(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) < self.robot_size and abs(pos1[1] - pos2[1]) < self.robot_size

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=self.state_size))
        model.add(layers.Conv2D(24, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
import gym
import numpy as np
from obstacle_avoidance_env import ObstacleAvoidanceEnv
from dqn_agent import DQNAgent

env = ObstacleAvoidanceEnv()
state_size = env.observation_space.shape
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

for e in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, *state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{1000}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

env.close()
