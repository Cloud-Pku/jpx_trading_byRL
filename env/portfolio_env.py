from __future__ import annotations

import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import logging
from finquant.portfolio import build_portfolio
import time
import pickle
matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class StockPortfolioEnvJpx(gym.Env):
    """portfolio allocation environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date
    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
        run_whole_train=False,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.init_day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.run_whole_train = run_whole_train
        self.render_index = 0
        if self.run_whole_train:
            self.lookback = len(self.df['date'].unique())
            print("+"*50,self.df['date'].unique())
            

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space, ))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        # self.covs = self.data["cov_list"].values[0]
        self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list]).ravel('F')
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state: inital portfolio return + individual stock return + individual weights
        self.sharpe_ratio = 1.

        # memorize portfolio value each step
        self.sharpe_ratio_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= self.init_day + self.lookback - 1
        # if True in np.isnan(actions):
        #     print(self.day, type(actions[0]), len(actions))

        if self.run_whole_train:
            filename = "/mnt/lustre/chenyun/jpx_trading/example/results/whole_cumulative_reward_" + str(self.render_index) +".png"
        else:
            filename = "/mnt/lustre/chenyun/jpx_trading/example/results/cumulative_reward.png"
        if self.terminal:
            if self.run_whole_train:
                self.render_index += 1
                print("!"*40, self.render_index)
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot( df.daily_return.cumsum(), "r")
            plt.xticks(range(len(self.date_memory)), self.date_memory, rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Cumulative Rewards')
            plt.title('Cumulative Rewards in Days')
            if self.run_whole_train:
                plt.axvline(947)
            plt.savefig(filename)
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.xticks(range(len(self.date_memory)), self.date_memory, rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Rewards')
            plt.title('Rewards in Days')
            plt.savefig("/mnt/lustre/chenyun/jpx_trading/example/results/rewards.png")
            plt.close()
            print("="*40)
            print(f"final_sharpe_ratio:{self.sharpe_ratio}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (df_daily_return["daily_return"].mean() / df_daily_return["daily_return"].std())
                print("Sharpe: ", sharpe)

            if self.date_memory[-1] == "2021-12-02":
                with open("/mnt/lustre/chenyun/jpx_trading/example/log/sharpe_ratio.pkl", 'rb') as f:
                    sharpe = pickle.load(f)
                    sharpe.append(self.sharpe_ratio)
                with open("/mnt/lustre/chenyun/jpx_trading/example/log/sharpe_ratio.pkl", 'wb') as f:
                    pickle.dump(sharpe, f)
                return self.state, self.reward, self.terminal, {'sharpe_ratio' : self.sharpe_ratio}
            return self.state, self.reward, self.terminal, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            # weights = self.softmax_normalization(actions)
            weights = actions
            # print("="*30)
            # print(weights)
            # print(len(weights))
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data
            # print(last_day_memory)
            # print(last_day_memory.columns)

            # 

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list]).ravel('F')
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            #print("weights:", weights)

            # jpx return
            self.data.insert(loc = len(self.data.columns), column = 'rank', value = weights)
            def _calc_spread_return_per_day(one_day_data, portfolio_size: int = 200, toprank_weight_ratio: float = 2):
                """
                Args:
                    df (pd.DataFrame): predicted results
                    portfolio_size (int): # of equities to buy/sell
                    toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
                Returns:
                    (float): spread return
                """
                weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
                purchase = (one_day_data.sort_values(by='rank')['target'][:portfolio_size] * weights).sum() / weights.mean()
                short = (one_day_data.sort_values(by='rank', ascending=False)['target'][:portfolio_size] * weights).sum() / weights.mean()
                return purchase - short


            portfolio_return = _calc_spread_return_per_day(self.data)
            #print("portfolio_return:", portfolio_return)
            # if portfolio_return > 0.1:
            #     print("="*30)
            #     print(portfolio_return)
            #     print(self.day)
            #     print(self.df.loc[self.day-2])
            #     print(self.df.loc[self.day-1])
            #     print(self.df.loc[self.day])
            #     print(self.df.loc[self.day+1])
            # update portfolio value
            self.portfolio_return_memory.append(portfolio_return)
            # self.reward = np.mean(self.portfolio_return_memory) / np.std(self.portfolio_return_memory) - self.sharpe_ratio
            self.reward = portfolio_return
            self.sharpe_ratio = np.mean(self.portfolio_return_memory) / np.std(self.portfolio_return_memory)
            # save into memory
            
            self.date_memory.append(self.data.date.unique()[0])
            self.sharpe_ratio_memory.append(self.sharpe_ratio)

            # the reward is the new portfolio value or end portfolo value
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        # print("="*40)
        # print(len(self.df.date.unique()))
        if not self.run_whole_train:
            self.init_day = self.day = np.random.randint(0, len(self.df.date.unique())-251)
        else :
            self.init_day = self.day = 0
        # print(self.day)
        self.data = self.df.loc[self.day, :]
        # load states
        self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list]).ravel('F')
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({"date": date_list, "daily_return": portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_baseline(self):
        # initalize comparison benchmark
        self.benchmark = build_portfolio(
            names=list(self.data.tic.values),
            start_date=self.df.date[self.df.date.index[0]].unique()[0],
            end_date=self.df.date[self.df.date.index[-1]].unique()[0],
            data_api="yfinance"
        )
