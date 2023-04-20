import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
action_dim = 3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.actor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.Tanh(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.Tanh(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1),
                nn.Flatten(),
                nn.Linear(49, action_dim),
                nn.Softmax(dim=-1)
            )
        self.critic = nn.Sequential(
                            nn.Conv2d(1, 16, kernel_size=8, stride=4),
                            nn.Tanh(),
                            nn.Conv2d(16, 5, kernel_size=2, stride=4),
                            # nn.Tanh(),
                            nn.Conv2d(5, 1, kernel_size=5, stride=1),
                            # nn.Linear(7, 7),
                            # nn.Tanh(),
                            # nn.Linear(1, 1),
                            nn.Flatten(0),
                            nn.Softmax(dim=-1)
                        )
        self.critic.float()
    def forward(self, x):
        for layer in self.critic:
            x = layer(x)
            print(x.size())
        return x


model = Model()
x = torch.randn(1000,1,84,84)

# Let's print it
model(x)