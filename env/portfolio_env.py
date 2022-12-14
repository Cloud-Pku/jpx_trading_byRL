from __future__ import annotations

import os
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
        stock_num : int
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
        exp_name,
        df,
        stock_num,
        feature_ws,
        reward_scaling,
        tech_indicator_list,
        lookback=252,
        day=0,
        run_whole_train=False,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.exp_name = exp_name
        self.day = day
        self.init_day = day
        self.lookback = lookback
        self.df = df
        self.tic = self.df.tic.unique()
        self.stock_num = stock_num
        self.feature_ws = feature_ws
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        self.render_index = 0
        self.run_whole_train = run_whole_train
        if self.run_whole_train:
            self.lookback = len(self.df.date.unique()) - max(self.feature_ws)
        
        self.fig_path = os.path.abspath(os.path.join(os.getcwd(), self.exp_name,"fig")) 
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

        # action_space normalization and shape is self.stock_num
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_num, ))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stock_num, sum(self.feature_ws)),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        # # self.covs = self.data["cov_list"].values[0]
        # self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list]).ravel('F')
        self.terminal = False

        # initalize state: inital portfolio return + individual stock return + individual weights

        # memorize portfolio value each step
        self.sharpe_ratio_memory = []
        # memorize portfolio return each step
        self.portfolio_return_memory = []
        self.actions_memory = []
        self.date_memory = []
        self.rankIC_memory = []


    def cal_state(self):
        curr_state = []
        for tic in self.tic:
            tmp_tic = self.df[self.df.tic == tic]
            tmp_feature = []
            for i, tech in enumerate(self.tech_indicator_list):
                tmp_feature.append(tmp_tic[tech].values[self.day - self.feature_ws[i] + 1: self.day + 1])
            curr_state.append(np.hstack(tmp_feature))
        self.state = np.vstack(curr_state)
        if self.state.shape != (100, 154):
            print(self.state.shape)
            print(self.day)
            print(self.feature_ws)
            logging.info('state shape({})\tday({})'.format(self.state.shape, self.day))
            logging.info('Init Day({})\tLookback({})\tEnd Day({})'.format(self.init_day, self.lookback, self.init_day + self.lookback - 1))
        
    def _calc_spread_return_per_day(self, one_day_data, portfolio_size: int = 10, toprank_weight_ratio: float = 2):
            """
            calcualte portfolio return
            Args:
                df (pd.DataFrame): predicted results
                portfolio_size (int): # of equities to buy/sell
                toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
            Returns:
                (float): spread return
            """
            weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
            purchase = (one_day_data.sort_values(by='rank', ascending=False)['target'].iloc[:portfolio_size] * weights).sum() / weights.mean()
            short = (one_day_data.sort_values(by='rank')['target'].iloc[:portfolio_size] * weights).sum() / weights.mean()
            return purchase - short

    def step(self, actions):

        # print(self.day, self.init_day + self.lookback - 1)
        self.terminal = self.day >= self.init_day + self.lookback - 2


        # ======================
        weights = actions
        
        self.actions_memory.append(weights)
        
        self.data = self.df.loc[self.day, :]
        
        self.data.insert(loc = len(self.data.columns), column = 'rank', value = weights)
        
        # calculate rank IC
        tmp_rank = np.argsort(weights)
        exp_rank = np.argsort(self.data['target'].tolist())
        rankIC = sum([(tmp_rank[i] - exp_rank[i])**2 for i in range(self.stock_num)]) * 6 / (self.stock_num - self.stock_num**3) + 1
        self.rankIC_memory.append(rankIC)
        # jpx return
        portfolio_return = self._calc_spread_return_per_day(self.data)
        self.portfolio_return_memory.append(portfolio_return)
        # calculate sharpe ratio
        sharpe_ratio = np.mean(self.portfolio_return_memory) / (np.std(self.portfolio_return_memory) + 1e-7)
        self.date_memory.append(self.data.date.unique()[0])
        self.sharpe_ratio_memory.append(sharpe_ratio)
        # define reward
        self.reward = rankIC
        # load next state
        self.day += 1
        self.cal_state()
        # =====================
            
        if self.run_whole_train:
            cum_reward_file = self.fig_path +"/whole_cumulative_reward_" + str(self.render_index) +".png"
            sharpe_file = self.fig_path +"/whole_sharpe_ratio_" + str(self.render_index) +".png"
            rankIC_file = self.fig_path +"/whole_rankIC_" + str(self.render_index) +".png"
            cum_rankIC_file = self.fig_path +"/whole_cumulative_rankIC_" + str(self.render_index) +".png"
        else:
            cum_reward_file = self.fig_path +"/cumulative_reward.png"
            sharpe_file = self.fig_path +"/sharpe_ratio.png"
            rankIC_file = self.fig_path +"/rankIC.png"
            cum_rankIC_file = self.fig_path +"/cumulative_rankIC.png"
        if self.terminal: # change
            if self.run_whole_train:
                self.render_index += 1
                print("="*10, self.render_index, "="*10)
                
            # render rankIC
            plt.plot(self.rankIC_memory, "r")
            plt.xticks(range(len(self.date_memory)), self.date_memory, rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Rank IC')
            plt.title('Rank IC in Days')
            plt.axhline(np.mean(self.rankIC_memory))
            plt.text(0, np.mean(self.rankIC_memory), \
                str(np.mean(self.rankIC_memory)), fontsize=7)
            if self.run_whole_train:
                plt.axvline(947)
            plt.savefig(rankIC_file)
            plt.close()
            
            # render cum rankIC
            df = pd.DataFrame(self.rankIC_memory)
            df.columns = ["rankIC"]
            plt.plot( df.rankIC.cumsum(), "r")
            plt.xticks(range(len(self.date_memory)), self.date_memory, rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Cumulative Rank IC')
            plt.title('Cumulative Rank IC in Days')
            if self.run_whole_train:
                plt.axvline(947)
            plt.savefig(cum_rankIC_file)
            plt.close()

            # render cum reward
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
            plt.savefig(cum_reward_file)
            plt.close()
            
            # render sharpe ratio
            plt.plot(self.sharpe_ratio_memory[1:], "r")
            plt.xticks(range(len(self.date_memory)-1), self.date_memory[1:], rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Sharpe Ratio')
            plt.title('Sharpe Ratio in Days')
            plt.text(len(self.date_memory)-1, self.sharpe_ratio_memory[-1], \
                str(self.sharpe_ratio_memory[-1]), fontsize=7)
            if self.run_whole_train:
                plt.axvline(947)
            plt.savefig(sharpe_file)
            plt.close()
            
            # render rewards
            plt.plot(self.portfolio_return_memory, "r")
            plt.xticks(range(len(self.date_memory)), self.date_memory, rotation=45)
            ax=plt.gca()
            x_major_locator=MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=6)
            plt.xlabel('Days')
            plt.ylabel('Rewards')
            plt.title('Rewards in Days')
            plt.savefig(self.fig_path +"/rewards.png")
            plt.close()
            print("="*40)
            print(f"mean_rank_IC:{np.mean(self.rankIC_memory)}")
            print(f"final_sharpe_ratio:{sharpe_ratio}")
            print(f"portfolio_return:{sum(self.portfolio_return_memory)}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            
            if self.run_whole_train:
                print("="*10, 'final_eval_reward:', np.sum(self.portfolio_return_memory), "="*10)
            return self.state, self.reward, self.terminal, {}
        else:
            return self.state, self.reward, self.terminal, {}


    def reset(self):
        # print("="*40)
        # print(len(self.df.date.unique()))
        if not self.run_whole_train:
            self.init_day = self.day = np.random.randint(max(self.feature_ws), len(self.df.date.unique())- self.lookback +1)
        else :
            self.init_day = self.day = max(self.feature_ws) 
        self.data = self.df.loc[self.day, :]
        self.cal_state()
        # load states
        self.terminal = False
        self.rankIC_memory = []
        self.portfolio_return_memory = []
        self.sharpe_ratio_memory = []
        self.actions_memory = []
        self.date_memory = []
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
