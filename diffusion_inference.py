import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet1DModel
import torch.optim as optim

# load data
data_path = './data/data.csv'
df = pd.read_csv(data_path)

def parse_array(s):
    return np.array(eval(s), dtype=np.float32)

r = np.stack(df['r'].apply(parse_array))
alpha = np.stack(df['alpha'].apply(parse_array))
beta = np.stack(df['beta'].apply(parse_array))
x = np.stack(df['x'].apply(parse_array))

# 200 for testing
test_r, test_alpha, test_beta, test_x = r[1500:], alpha[1500:], beta[1500:], x[1500:]

# dataset definition
class PortfolioDataset(Dataset):
    def __init__(self, r, alpha, beta, x):
        self.r = torch.tensor(r, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        target = self.x[idx]
        condition = torch.cat([target, self.r[idx], self.alpha[idx], self.beta[idx]])
        # print(condition.shape) # torch.Size([15])
        # print(target.shape)    # torch.Size([5])
        return condition

# dataset
dataset = PortfolioDataset(test_r, test_alpha, test_beta, test_x)

####################### config for parameters #######################
num_steps = 100 # steps for sampling time
 
# beta
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
 
# alpha, alpha_prod, alpha_prod_previous, alpha_bar_sqrt
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
####################### config for parameters #######################


####################### MLP for x[t] + t --> noise_predict #######################
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
 
        self.linears = nn.ModuleList(
            [
                nn.Linear(20, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 20),
            ]
        )
        
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
 
    def forward(self, x, t):
        #  x = x[0]
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x
####################### MLP for x[t] + t --> noise_predict #######################

####################### at xt, sample t-1 to construct x(t-1) #######################
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    ## 1. solve bar_u_t
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    # put into the unet, obtain the predicted noise at t: eps_theta
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
 
    ## 2. solve x[t-1]
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)
####################### at xt, sample t-1 to construct x(t-1) #######################

####################### iteration loop to construct x0 from xt #######################
def p_sample_loop(model, noise_x_t, n_steps, betas, one_minus_alphas_bar_sqrt):
    # obtain noise xt
    cur_x = noise_x_t
    x_seq = [noise_x_t]
    # recover x(t-1), ..., x(0) from xt
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq, cur_x # x_seq: xt, ..., x0; cur_x: x0
####################### iteration loop to construct x0 from xt #######################

# 1. load the trained diffusion model
model = MLPDiffusion(num_steps)
model.load_state_dict(torch.load('./model/diffusion/ddpm_900.pth'))
# 2. generate the random noise xt
# print(dataset)
noise_x_t = torch.randn(torch.Size([200, 20]))
# 3. inverse diffusion into x(t-1), ..., x0
x_seq, cur_x = p_sample_loop(model, noise_x_t, num_steps, betas, one_minus_alphas_bar_sqrt)
print(cur_x)