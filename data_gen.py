import numpy as np
import os
import pandas as pd

def generate_sample(n_assets, B):
    r = np.random.uniform(0.05, 0.55, n_assets)  # income rate
    alpha = np.random.uniform(0.05, 0.15, n_assets)  # lowest bound
    beta = np.random.uniform(0.35, 0.55, n_assets)  # highest bound

    # Generate optimal solutions using linear programming
    from scipy.optimize import linprog
    c = -r  # Maximize the gain (negative sign turns into a minimization problem)
    A_ub = np.vstack([
        np.ones(n_assets),  # Budget Constraints
        -np.eye(n_assets),  # Minimum investment ratio
        np.eye(n_assets)    # Maximum investment ratio
    ])
    b_ub = np.hstack([
        B,             # Total Budget
        -alpha * B,    # Minimum investment amount per asset
        beta * B       # Maximum investment amount per asset
    ])
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if result.success:
        return r, alpha, beta, result.x  # Returns the generated sample data
    else:
        raise ValueError("Sample generation failed")

#######################config begin#######################
n_samples = 1700
n_assets = 5
B = 100
#######################config  end #######################

samples = []
os.makedirs('./data', exist_ok=True)

for _ in range(n_samples):
    samples.append(generate_sample(n_assets, B))
    print(generate_sample(n_assets, B))

data_list = []
for sample in samples:
    r, alpha, beta, x = sample
    data_list.append({
        "r": r.tolist(),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "x": x.tolist()
    })

df = pd.DataFrame(data_list)
df.to_csv('./data/data.csv', index=False)