from easydict import EasyDict
from data.utils import FeatureEngineer, data_split, load_dataset
import pickle
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import random


def generate_config(exp_name):
    
    trading_ppo_config = dict(
        feature_ws=[30, 30, 30, 30, 30, 1, 1, 1, 1],
        exp_name=exp_name,
        env=dict(
            collector_env_num=8,
            evaluator_env_num=1,
            n_evaluator_episode=1,
            stop_value=10000,
        ),
        policy=dict(
            cuda=False,
            action_space='continuous',
            model=dict(
                obs_shape=[100, 154],
                action_shape=100,
                action_space='continuous',
                encoder_hidden_size_list=[512, 10, 2000, 128],
                critic_head_hidden_size=128,
                actor_head_hidden_size=128,
            ),
            learn=dict(
                epoch_per_collect=2,
                batch_size=64,
                learning_rate=0.001,
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.2,
                learner=dict(save_ckpt_after_iter=100
                ),
            ),
            collect=dict(
                n_sample=256,
                unroll_len=1,
                discount_factor=0.99,
                gae_lambda=0.95,
            ),
            eval=dict(evaluator=dict(eval_freq=100,),
            hook=dict()
            ),
        )
    )
    trading_ppo_config = EasyDict(trading_ppo_config)
    main_config = trading_ppo_config
    trading_ppo_create_config = dict(
        env=dict(
            type='cartpole',
            import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='ppo'),
    )
    trading_ppo_create_config = EasyDict(trading_ppo_create_config)
    create_config = trading_ppo_create_config
    return main_config, create_config