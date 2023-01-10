import os
import sys
sys.path.insert(0, "/mnt/lustre/chenyun/jpx_trading/jpx_trading_byRL")
from ditk import logging

from lighttuner.hpo import R, uniform, choice
from lighttuner.hpo import hpo
from lighttuner.scheduler import run_scheduler_local

def demo():
    dir_name = os.path.abspath('/mnt/lustre/chenyun/jpx_trading/jpx_trading_byRL')

    with run_scheduler_local(task_config_template_path=os.path.join(dir_name, "note.py"),
                             dijob_project_name="jpx_ppo_hpo") as scheduler:

        opt = hpo(scheduler.get_hpo_callable())
        cfg, ret, metrics = opt.grid() \
            .max_steps(5) \
            .max_workers(4) \
            .maximize(R['eval_value']) \
            .spaces({'feature_ws': choice([
                [10, 10, 10, 10, 10, 1, 1, 1, 1],
                [30, 30, 30, 30, 30, 1, 1, 1, 1],
                [20, 20, 20, 20, 20, 1, 1, 1, 1],
                ])}).run()
        print(cfg)
        print(ret)
        
if __name__ == "__main__":
    demo()