import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json


class SlidePuzzleEnv(gym.Env):
    def __init__(self):
        super(SlidePuzzleEnv, self).__init__()

        self.rows = 4
        self.cols = 5
        self.state_size = self.rows * self.cols

        self.observation_space = spaces.Box(
            low=0, high=self.state_size-1, shape=(self.state_size,), dtype=np.int32)
        # 4 acciones: arriba, abajo, izquierda, derecha
        self.action_space = spaces.Discrete(4)

        self.goal_state = np.array([[1, 2, 3, 4, 5],
                                    [6, 7, 8, 9, 10],
                                    [11, 12, 13, 14, 15],
                                    [16, 17, 18, 19, 0]], dtype=np.int32)

        self.start_state = np.array([[1, 7, 2, 4, 5],
                                     [6, 3, 0, 8, 9],
                                     [11, 12, 13, 15, 10],
                                     [16, 17, 18, 14, 19]], dtype=np.int32)

        self.state = self.start_state.copy()

    def reset(self):
        self.state = self.start_state.copy()
        self.steps_taken = 0
        return self.state.flatten()

    def step(self, action):
        empty_pos = np.argwhere(self.state == 0)[0]
        row, col = empty_pos
        reward = -1
        done = False

        if action == 0 and row > 0:  # Mover hacia arriba
            self.state[row, col], self.state[row-1,
                                             col] = self.state[row-1, col], self.state[row, col]
        elif action == 1 and row < self.rows - 1:  # Mover hacia abajo
            self.state[row, col], self.state[row+1,
                                             col] = self.state[row+1, col], self.state[row, col]
        elif action == 2 and col > 0:  # Mover hacia la izquierda
            self.state[row, col], self.state[row, col -
                                             1] = self.state[row, col-1], self.state[row, col]
        elif action == 3 and col < self.cols - 1:  # Mover hacia la derecha
            self.state[row, col], self.state[row, col +
                                             1] = self.state[row, col+1], self.state[row, col]
        else:
            reward = -10  # Penalización por movimiento inválido

        self.steps_taken += 1

        if np.array_equal(self.state, self.goal_state):
            reward = 100
            done = True
        elif self.steps_taken >= 200:
            reward = -10
            done = True

        return self.state.flatten(), reward, done, {}

    def render(self):
        print(self.state)


def save_q_table(q_table, filename):
    q_table_serializable = {str(k): list(v) for k, v in q_table.items()}
    with open(filename, 'w') as f:
        json.dump(q_table_serializable, f)


def train(episodes):
    env = SlidePuzzleEnv()

    q_table_greedy = {}
    q_table_epsilon_greedy = {}

    learning_rate = 0.1
    discount_factor = 0.9
    epsilon_epsilon_greedy = 1.0
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode_greedy = np.zeros(episodes)
    rewards_per_episode_epsilon_greedy = np.zeros(episodes)

    for i in range(episodes):
        state = tuple(env.reset())
        terminated = False
        total_reward_greedy = 0

        while not terminated:
            if state not in q_table_greedy:
                q_table_greedy[state] = np.zeros(env.action_space.n)

            action = np.argmax(q_table_greedy[state])
            new_state, reward, terminated, _ = env.step(action)
            new_state = tuple(new_state)

            if reward == 100:  # Si la recompensa es 100, el agente ha ganado
                total_reward_greedy = reward
            else:
                total_reward_greedy += reward

            if new_state not in q_table_greedy:
                q_table_greedy[new_state] = np.zeros(env.action_space.n)

            q_table_greedy[state][action] = q_table_greedy[state][action] + learning_rate * \
                (reward + discount_factor *
                 np.max(q_table_greedy[new_state]) - q_table_greedy[state][action])
            state = new_state

        rewards_per_episode_greedy[i] = total_reward_greedy

        state = tuple(env.reset())
        terminated = False
        total_reward_epsilon_greedy = 0

        while not terminated:
            if state not in q_table_epsilon_greedy:
                q_table_epsilon_greedy[state] = np.zeros(env.action_space.n)

            if rng.random() < epsilon_epsilon_greedy:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table_epsilon_greedy[state])

            new_state, reward, terminated, _ = env.step(action)
            new_state = tuple(new_state)

            if reward == 100:  # Si la recompensa es 100, el agente ha ganado
                total_reward_epsilon_greedy = reward
            else:
                total_reward_epsilon_greedy += reward

            if new_state not in q_table_epsilon_greedy:
                q_table_epsilon_greedy[new_state] = np.zeros(
                    env.action_space.n)

            q_table_epsilon_greedy[state][action] = q_table_epsilon_greedy[state][action] + learning_rate * \
                (reward + discount_factor *
                 np.max(q_table_epsilon_greedy[new_state]) - q_table_epsilon_greedy[state][action])
            state = new_state

        epsilon_epsilon_greedy = max(
            epsilon_epsilon_greedy - epsilon_decay_rate, 0)
        rewards_per_episode_epsilon_greedy[i] = total_reward_epsilon_greedy

        if (i + 1) % 100 == 0:
            print(
                f"Episodio {i + 1} - Recompensa Greedy: {rewards_per_episode_greedy[i]} - Recompensa Epsilon-Greedy: {rewards_per_episode_epsilon_greedy[i]}")

    env.close()
    print("Entrenamiento completado.")

    save_q_table(q_table_greedy, 'q_table_greedy.json')
    save_q_table(q_table_epsilon_greedy, 'q_table_epsilon_greedy.json')

    avg_rewards_greedy = np.zeros(episodes // 100)
    avg_rewards_epsilon_greedy = np.zeros(episodes // 100)
    for t in range(episodes // 100):
        avg_rewards_greedy[t] = np.mean(
            rewards_per_episode_greedy[t * 100:(t + 1) * 100])
        avg_rewards_epsilon_greedy[t] = np.mean(
            rewards_per_episode_epsilon_greedy[t * 100:(t + 1) * 100])

    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards_greedy, label='Greedy')
    plt.plot(avg_rewards_epsilon_greedy, label='Epsilon-Greedy')
    plt.xlabel('Bloques de 100 episodios')
    plt.ylabel('Recompensa media')
    plt.legend()
    plt.title('Comparación de políticas Greedy y Epsilon-Greedy')
    plt.show()


if __name__ == "__main__":
    train(10000)
