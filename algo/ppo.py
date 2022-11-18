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




def render_whole_train():
    def _rwt(ctx):
        if ctx.total_step % 100 == 0:
            print(ctx.total_step)
            os.rename("/home/PJLAB/chenyun/jpx_trading/PortfolioOpt-main/main/results/whole_cumulative_reward.png",
                "/home/PJLAB/chenyun/jpx_trading/PortfolioOpt-main/main/results/whole_cumulative_reward_"+ str(ctx.total_step) + ".png")
    return _rwt

def trainer(env_name, main_config, create_config, env_train_kwargs, env_test_kwargs, run_whole_train_kwargs, max_train_iter):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **env_train_kwargs))
                for _ in range(cfg.env.collector_env_num)
            ],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **env_test_kwargs))
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager
        )

        run_whole_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **run_whole_train_kwargs)) for _ in range(1)
            ],
            cfg=cfg.env.manager
        )
        print(env_test_kwargs)

        cfg.seed = 5
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        print("="*40, "cfg.policy.model")
        print(cfg.policy.model)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        # task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(interaction_evaluator(cfg, policy.eval_mode, run_whole_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=10000))
        task.use(termination_checker(max_train_iter=max_train_iter))
        # task.use(render_whole_train())
        task.run()

def tester(env_name, main_config, create_config, env_train_kwargs, env_test_kwargs, run_whole_train_kwargs, max_train_iter):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **run_whole_train_kwargs))
                for _ in range(cfg.env.collector_env_num)
            ],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **run_whole_train_kwargs))
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        print("="*40, "cfg.policy")
        print(cfg.policy)
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.run()
