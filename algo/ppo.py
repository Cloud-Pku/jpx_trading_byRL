import gym
import sys, os
sys.path.insert(0, os.getcwd())

from model.vac import VAC
from ditk import logging
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, gae_estimator, termination_checker
from ding.utils import set_pkg_seed




def log_step():
    def _rwt(ctx):
        print("="*10, "train_iter:",ctx.train_iter, "env_step:",ctx.env_step, "="*10)
    return _rwt

def trainer(env_name, main_config, create_config, env_train_kwargs, run_whole_train_kwargs, max_train_iter = 15000, seed=0):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, seed=seed)
    
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **env_train_kwargs))
                for _ in range(cfg.env.collector_env_num)
            ],
            cfg=cfg.env.manager
        )

        run_whole_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **run_whole_train_kwargs)) for _ in range(1)
            ],
            cfg=cfg.env.manager
        )
        
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        print("="*10,"cuda used", "="*10)
        print(cfg.policy.cuda)
        print("="*10,"seed", "="*10)
        print(cfg.seed)
        print("="*25)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        # task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(interaction_evaluator(cfg, policy.eval_mode, run_whole_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=10000))
        task.use(termination_checker(max_env_step=1e7 ,max_train_iter=max_train_iter))
        task.use(log_step())
        task.run()
