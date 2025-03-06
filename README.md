# Diffusion-Based Portfolio Optimization

## Overview
This is an initial test for using diffusion to solve optimization problems. Specifically, this project aims to use diffusion to address a simple portfolio optimization problem. The problem is formulated as follows:
$$
\begin{aligned}
\min \quad & -r^T x \\
\text{s.t.} \quad 
& \sum_i x_i = 1 \\
& \alpha \leq x_i \leq \beta \\
& r_i \geq 0 \\
& 0 \leq \alpha \leq \beta \leq 1,
\end{aligned}
$$
where the varible $x_i$ represents the investment proportion of the $i$-th stock, the parameters $r_i$ represents the investment return ratio of the $i$-th stock, $\alpha$ is the lowest bound of each $x_i$, and $\beta$ is the highest bound of $x_i$.

Firstly I generated some ground truth data ($r$, $\alpha$, $\beta$, $x$), and selected some of them as training data, while others as testing data.

Secondly I trained a simple fully connect model and test its accuracy as a baseline. 

Then for the most important part, I created a DDPM processing using UNet, and train the model with the input of UNet being ($r$, $\alpha$, $\beta$, $x$). However, I found trouble at the inference step, since I required $x$ with given $r$, $\alpha$ and $\beta$, but this seems difficult because I treated $r$, $\alpha$, $\beta$ and $x$ as a whole during training.


## Project Structure
```
diffusion_opt/
│-- data/                  
|   │-- data.csv 
│-- model/                 
|   |-- diffusion/
|   |-- fc_model.pth
│-- data_gen.py              
│-- full_connect.py          
│-- fc_test.py               
│-- diffusion.py             
│-- diffusion_inference.py   
│-- README.md                
```

`data/` : Stores generated financial data.

`model/` : Stores trained models.

`data_gen.py` : Generates $r$, $\alpha$, $\beta$ and solve $x$ using linprog.

`full_connect.py` : Trains a fully connected model for baseline.

`fc_test.py` : Tests the trained FCN on new data.

`diffusion.py` : Trains a DDPM model for the portfolio problem.

`diffusion_inference.py` : Runs inference using the trained diffusion model. (**Meet Trouble Here**)

## Trouble
Just as I have mentioned in the Overview, I trained the DDPM model with the input bing ($r$, $\alpha$, $\beta$, $x$), and for the inference part, I needed to obtain $x$ with fixed $r$, $\alpha$ and $\beta$, which is really difficult.

So, I want to know how to allocate the parameters $r$, $\alpha$, $\beta$ and the variable $x$ in the input to obtain a reasonable diffusion model.