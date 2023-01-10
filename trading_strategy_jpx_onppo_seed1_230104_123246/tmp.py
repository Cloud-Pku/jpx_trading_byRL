import torch
from model.vac import VAC
from formatted_total_config import main_config
cfg = main_config
model = VAC(**cfg.policy.model)
net = torch.load("./ckpt/eval.pth.tar")
x = torch.rand( 3,100, 194)
print(x)
# print(list(net.parameters()))
print(net(x))