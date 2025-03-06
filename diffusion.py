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

# 1500 for training
train_r, train_alpha, train_beta, train_x = r[:1500], alpha[:1500], beta[:1500], x[:1500]

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

# dataloader
dataset = PortfolioDataset(train_r, train_alpha, train_beta, train_x)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

####################### x[t]: x[0] + t --> x[t] #######################
def q_x(x_0, t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    x_t = alphas_t * x_0 + alphas_1_m_t * noise
    return x_t
####################### x[t]: x[0] + t --> x[t] #######################

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

####################### loss(real noise 'eps', predicted noise) #######################
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """sample t to compute loss"""
    batch_size = x_0.shape[0]
 
    # t.shape = torch.Size([batchsize, 1])
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)
 
    ## 1. obtain xt
    # coeff for x0
    a = alphas_bar_sqrt[t]
    # coeff for eps
    aml = one_minus_alphas_bar_sqrt[t]
    # random noise eps
    e = torch.randn_like(x_0)
    # xt = a*x0 + aml*eps
    x = x_0 * a + e * aml
 
    ## 2. xt into Unet, obtain predicted noise at timestep t
    output = model(x, t.squeeze(-1))
 
    ## 3. compute loss between real noise 'eps' and predicted noise 'noise_predict'
    # print(e.shape)
    # print(output.shape)
    loss = (e - output).square().mean()
    return loss
####################### loss(real noise 'eps', predicted noise) #######################


# training
print('Training model...')
batch_size = 64
num_epoch = 1000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

os.makedirs('./model/diffusion', exist_ok=True)
for t in range(num_epoch):
    for idx, batch_x in enumerate(dataloader):
        # print(f"idx: {idx}")
        # print(f"batch_x: {batch_x.shape}")
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
 
    if (t % 100 == 0):
        print(f"epoch {t}, loss = {loss}.")
        torch.save(model.state_dict(), './model/diffusion/ddpm_{}.pth'.format(t))