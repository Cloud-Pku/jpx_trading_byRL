import torch
from model.vac import VAC
from ding.policy import PPOPolicy
from trading_strategy_jpx_onppo_seed2.formatted_total_config import main_config
from env.portfolio_env import StockPortfolioEnvJpx
from note import run_whole_train_kwargs
import pickle
from global_data import *
cfg = main_config
model = VAC(**cfg.policy.model)
policy = PPOPolicy(cfg.policy, model=model)
policy.learn_mode.load_state_dict( torch.load("./trading_strategy_jpx_onppo_seed2/ckpt/eval.pth.tar", map_location=torch.device('cpu')))

# x = torch.rand( 3,100, 154)
# print(x)
# print(policy._eval_model.forward(x, mode='compute_actor'))
run_whole_train_kwargs['exp_name'] = 'test'
jpxenv = StockPortfolioEnvJpx(**run_whole_train_kwargs)

state = jpxenv.reset()
num = 0
label = []
while jpxenv.terminal is False:
    print(len(data_set.data_book))
    num += 1
    action = policy._eval_model.forward(torch.from_numpy(state).float(), mode='compute_actor')['action']
    action = action.squeeze()
    label.append(action.detach().numpy().tolist())
    state, reward, done, info = jpxenv.step(action.detach().numpy())

print(data_set.data_book)
with open("./test/data_set.pkl", 'wb') as f:
    pickle.dump(data_set.data_book, f)
with open("./test/label.pkl", 'wb') as f:
    pickle.dump(label, f)