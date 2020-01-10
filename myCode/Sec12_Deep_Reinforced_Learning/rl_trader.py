# Install and import TF2
# !pip install -q tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)

# More imports
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

# get the data
# !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/aapl_msi_sbux.csv

# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data():
  # returns a T x 3 list of stock prices
  # each row is a different stock
  # 0 = AAPL
  # 1 = MSI
  # 2 = SBUX
  df = pd.read_csv('aapl_msi_sbux.csv')
  return df.values

### The experience replay memory ###
class ReplayBuffer:
  def _init__(self,obs_dim,act_dim,size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32) # state
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32) # next state
    self.acts_buf = np.zeros(size, dtype=np.unit8) # actions
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.unit8)
    self.ptr, self.size, self.max_size = 0, 0, size # buffer pointer
  
  # store transition into buffer
  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  # sampling by selecting random indices
  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0,self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])

def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # note: you could also populate the replay buffer here

  states = []
  for _ in range(env, n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(direcory):
    os.makedirs(directory)


def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  # A multi-layer-perceptron #

  # input layer
  i = Input(shape=(input_dim,))
  x = i

  # hidden layers
  for _ in range(n_hidden_layers):
    x = Dense(hidden_dim, activation='relu')(x)

  # final layer
  x = Dense(n_action)(x)

  # make the model
  model = Model(i, x)

  model.compile(loss='mse', optimizer='adam')
  print(model.summary())
  return model


class MultiStockEnv:
  """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
      - # shares of stock1 owned
      - # shares of stock2 owned
      - # shares of stock3 owned
      - price of stock 1 (using daily close price)
      - price of stock 2 (using daily close price)
      - price of stock 3 (using daily close price)
      - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities
      - for each stock you can:
        - 0 = sell
        - 1 = hold
        - 2 - buy
  """
  ############# PUBLIC METHODS ################

  # data input is a 2D array with a timeseries of stock prices
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # ...
    # 0 = sell, 1 = hold, 2 = buy
    self.action_list = list(
        map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()

  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history(self.cur_step)
    self.cash_in_hand = self.initial_investment
    return self._get_obs()  # return state vector

  def step(self, action):
    assert action in self.action_space

    # get current value before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history(self.cur_step)

    # perform the trade
    self._trade(action)

    # get the new value after taking te action
    cur_val = self._get_val()

    # reward is the increase in portfolio value
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the OpenAI Gym API
    return self._get_obs(), reward, done, info

  ######## PRIVATE METHODS #############

  def _get_obs(self):        # get state, a vector of 3 components
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned  # list of size 3
    obs[self.nstock:2*self.n_stock] = self.stock_price  # value of the stock
    obs[-1] = self.cash_in_hand
    return obs

  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means
    # buy 1st stock
    # hold 2nd stock
    # sell 3rd stock
    action_vec = self.action_list[action]  # vector of size 3

    # determine which stocks to buy or sell
    sell_index = []  # stores index of stocks we want to sell
    buy_index = []  # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # Note: to simplify the problem. when we sell, we sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      # Note: when buying, we will loop through each stock we want to buy
      #       and buy one share at a time untiull we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1  # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False


class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size)

  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action-size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # return action

  def replay(self, batch_size=32):
    # first check if replay buffer contains enough data
    if self.memory.size < batch_size:
      return

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    # Calculate the tentative target: Q(s',a)
    target = rewards + self.gamma * \
        np.amax(self.model.predict(next_state), axis=1)

    # The value of terminal states in zero
    # so set the target to be the reward only
    target[done] = rewards[done]

    # target is 1D while predictions 2D (batch_size , num_of_actions) array
    # with the Keras API, the target (usually) must have the same
    # shape as the predictions
    # However, we only need to update the network for the actions
    # which were actually taken
    # we can accomplish this by setting the target to be equal to
    # the prediction for all values
    # then only change the targets for the actions taken
    # Q(s,a)
    target_full = self.model.predict(states)
    target_full[np.arange(batch_size), actions] = target  # (row,col)

    # Run one training step
    self.model.train_on_batch(states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)
    state = next_state

  return info('cur_val')

######### MAIN PROGRAM ##################


if __name__ == '__main__':

  # config
  models_folder = 'rl_trader_models'
  rewards_folder = 'rl_trader_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  data = get_data()
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgetn(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.h5')

  # play the game num_episodes times
  for e in range(num_episodes):
    t0 = datetime.now()
    val = play_one_episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode and value: {val:.2f},duration: {dt}")
    portfolio_value.append(val)  # append episode end portfolio value

  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{model_folder}/dqn.h5')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

  # save portfolio value  for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
