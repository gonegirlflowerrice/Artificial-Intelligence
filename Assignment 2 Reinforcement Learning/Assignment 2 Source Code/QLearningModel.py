import argparse
import random
import numpy as np
from collections import defaultdict
from Environment import TreasureCube
import matplotlib.pyplot as plt

rewardList = []
q_table = {'000': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '001': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '002': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '003': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '010': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '011': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '012': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '013': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '020': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '021': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '022': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '023': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '030': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '031': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '032': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},
           '033': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0,'up': 0, 'down': 0},

           '100': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '101': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '102': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '103': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '110': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '111': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '112': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '113': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '120': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '121': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '122': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '123': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '130': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '131': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '132': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '133': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},

           '200': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '201': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '202': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '203': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '210': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '211': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '212': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '213': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '220': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '221': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '222': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '223': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '230': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '231': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '232': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '233': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},

           '300': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '301': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '302': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '303': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '310': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '311': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '312': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '313': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '320': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '321': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '322': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '323': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '330': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '331': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '332': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0},
           '333': {'left': 0, 'right': 0, 'forward': 0, 'backward': 0, 'up': 0, 'down': 0}}


# you need to implement your agent based on one RL algorithm
class QAgent(object):

    def __init__(self):
        learning_rate = 0.5
        discount_rate = 0.99
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.action_space = ['right', 'left', 'up', 'down', 'forward', 'backward']      # in TreasureCube
        # Create an array for action-value estimates and initialize it to zero.
        # self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))

    def take_action(self, state):
        epsilon = 0.01          # set exploration_rate
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < epsilon:
            return random.choice(self.action_space)             # Explore and take random action
        else:
            max_q = 0
            direction = random.choice(self.action_space)
            for key, value in q_table[state].items():
                if q_table[state][key] >= max_q:
                    max_q = q_table[state][key]
                    direction = key
            return direction

# implement your train/update function to update self.V or self.Q
# you should pass arguments to the train function

    def train(self, state, action, next_state, reward):
        if next_state == '333':
            q_table[state][action] = reward

        else:  # Improve Q-values with Bellman Equation.
            max_q = 0
            for key, value in q_table[next_state].items():
                if q_table[next_state][key] > max_q:
                    max_q = q_table[next_state][key]
            old_value = q_table[state][action]
            new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_rate * max_q)
            q_table[state][action] = new_value


def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = QAgent()

    for episode_num in range(0, max_episode):
        state = env.reset()                     # starting position
        terminate = False                       # Terminate = true when [3,3,3] is reached
        t = 0                                   # Initialise number of steps = 0
        episode_reward = 0                      # Initialise episode reward = 0
        while not terminate:                    # Run until the end of the episode
            action = agent.take_action(state)   # Select action
            reward, terminate, next_state = env.step(action)    # Take action
            episode_reward += reward                            # Updating episode reward
            #   env.render()
            #   print(f'step: {t}, action: {action}, reward: {reward}')
            print(action)
            t += 1                                              # Number of steps
            agent.train(state, action, next_state, reward)      # Update Q-table for Q(s,a)
            state = next_state                                   # Update state
        rewardList.append(episode_reward)  # Update statistics: Append episode reward to List q
        print(f'episode: {episode_num}, total_steps: {t} episode reward: {episode_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)


plt.plot(rewardList)
plt.show()