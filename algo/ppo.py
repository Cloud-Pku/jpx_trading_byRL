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
from typing import Callable
import pickle



def log_step():
    def _rwt(ctx):
        logging.info('Evaluation: Train Iter({})\tEnv Step({})\tEval Value({:.3f})'.format(ctx.train_iter, ctx.env_step, ctx.eval_value))
    return _rwt

def keep_eval():
    def _keep_eval(ctx):
        ctx.keep('env_step', 'env_episode', 'train_iter', 'last_eval_iter', 'eval_value')
    return _keep_eval

def final_ctx_saver(name: str) -> Callable:

    def _save(ctx: "Context"):
        if task.finish:
            with open(os.path.join(name, 'result.pkl'), 'wb') as f:
                final_data = {
                    'total_step': ctx.total_step,
                    'train_iter': ctx.train_iter,
                    'eval_value': ctx.eval_value,
                }
                if ctx.has_attr('env_step'):
                    final_data['env_step'] = ctx.env_step
                    final_data['env_episode'] = ctx.env_episode
                pickle.dump(final_data, f)

    return _save

def trainer(env_name, main_config, create_config, env_train_kwargs, run_whole_train_kwargs, max_train_iter = 1e7, seed=0):
    logging.getLogger().setLevel(logging.INFO)

    cfg = compile_config(main_config, create_cfg=create_config, auto=True, seed=seed)
    env_train_kwargs["exp_name"] = cfg.exp_name
    run_whole_train_kwargs["exp_name"] = cfg.exp_name
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
                lambda: DingEnvWrapper(gym.make("{}".format(env_name), **run_whole_train_kwargs))
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager
        )
        
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        logging.info("======= cuda =======")
        logging.info(cfg.policy.cuda)
        logging.info("======= seed =======")
        logging.info(cfg.seed)
        logging.info("======= exp_name =======")
        logging.info(cfg.exp_name)
        logging.info("====================")
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        task.use(keep_eval())
        # task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(interaction_evaluator(cfg, policy.eval_mode, run_whole_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(cfg, policy.learn_mode))
        task.use(log_step())
        task.use(termination_checker(max_env_step=1e7 ,max_train_iter=max_train_iter))
        task.use(final_ctx_saver(cfg.exp_name))
        task.use(CkptSaver(cfg, policy, train_freq=10000))
        task.run()
