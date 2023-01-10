import torch
from ding.utils import SequenceType

class StockRepresent(torch.nn.Module):
    def __init__(self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [512, 10, 2000, 128],
        ) -> None:
        super(StockRepresent, self).__init__()
        self.stock_num = obs_shape[0]
        self.linear1 = torch.nn.Linear(obs_shape[1], hidden_size_list[0])
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.layer_norm = torch.nn.LayerNorm(self.stock_num, eps=1e-6)
    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        x = self.linear2(self.relu(self.linear1(x)))
        x = x.reshape(-1, self.stock_num, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x
        
class MLP(torch.nn.Module):
    def __init__(self,
            obs_shape: SequenceType,
            hidden_size_list: SequenceType = [512, 10, 2000, 128],
        ) -> None:
        super(MLP, self).__init__()
        self.stock_num = obs_shape[0]
        self.co_feature = self.stock_num * hidden_size_list[1]
        # self.flatten = torch.nn.Flatten()
        self.represent = StockRepresent(obs_shape, hidden_size_list)
        self.relu = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(self.stock_num * hidden_size_list[1], hidden_size_list[2])
        self.linear4 = torch.nn.Linear(hidden_size_list[2], hidden_size_list[3])

        
    # def forward(self, x):
    #     x = x.permute(1,0,2)
    #     res = [self.linear2(self.relu(
    #         self.linear1(x[i])))
    #            for i in range(x.shape[0])]
    #     res = torch.stack(res)
    #     res = res.permute(1,0,2)
    #     res = res.reshape(res.shape[0], -1)
    #     res = self.linear4(self.relu(self.linear3(self.relu(res))))
    #     return res

    def forward(self, x):
        # x = x.unsqueeze(0)
        # x = x.permute(1,0,2)

        x = self.represent(x)
        x = x.reshape(-1, self.co_feature)
        # res = res.permute(1,0,2)
        res = self.linear4(self.relu(self.linear3(self.relu(x))))
        return res
    
net = MLP(obs_shape=[5,4])

# x = torch.Tensor([[[1.]*4]*3]*2)

x = torch.rand( 3,5, 4)
print(x)
# print(list(net.parameters()))
print(net(x))

torch.save(net.state_dict(), "./op.pkl")