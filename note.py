import sys
import os
sys.path.insert(0, os.getcwd())
from data.utils import FeatureEngineer, data_split, load_dataset
import pickle
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import random

with open("/mnt/lustre/chenyun/jpx_trading/jpx_trading_byRL/data/data_scale.pkl", 'rb') as f:
    df = pickle.load(f)
df = df.sort_values(['date', 'tic']).reset_index(drop=True)

train = data_split(df, '2017-01-05', '2020-11-22')
test = data_split(df, '2020-11-22', '2021-12-03')

df.index = df['date'].factorize()[0]
whole_data = df

print(whole_data)

# Process your data here [doing data  cleaning, features engineering here]

feature_list = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_30', 'cci_30', 'dx_30']
feature_ws = [30, 30, 30, 30, 30, 1, 1, 1, 1]
stock_num = len(train.tic.unique())
stock_feature_dimension = len(feature_list)
state_space = stock_num * sum(feature_ws)
print(f"Stock Feature Dimension: {stock_feature_dimension}, State Space: {state_space}")
feature_dimension = len(feature_list)
print(f"Feature Dimension: {feature_dimension}")
seed_num = 98
exp_name = 'trading_strategy_jpx_onppo_seed'+ str(seed_num)
env_train_kwargs = {
    'exp_name': exp_name,
    'seed_num': seed_num,
    'df': train,
    "state_space": state_space,
    "feature_ws": feature_ws,
    "stock_num": stock_num,
    "tech_indicator_list": feature_list,
    "action_space": stock_num,
    "reward_scaling": 1e-1
}

run_whole_train_kwargs = {
    'exp_name': exp_name,
    'seed_num': seed_num,
    'df': whole_data,
    "state_space": state_space,
    "feature_ws": feature_ws,
    "stock_num": stock_num,
    "tech_indicator_list": feature_list,
    "action_space": stock_num,
    "reward_scaling": 1e-1,
    "run_whole_train": True,
}

from gym.envs.registration import register
from algo.ppo import trainer
from config.trading_config import main_config, create_config
main_config.exp_name=exp_name

register(id='trading-v2', entry_point='env.portfolio_env:StockPortfolioEnvJpx', max_episode_steps=10000)
trainer('trading-v2', main_config, create_config, env_train_kwargs, run_whole_train_kwargs, seed=seed_num)
