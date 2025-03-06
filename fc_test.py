import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# config
DATA_PATH = './data/data.csv'
MODEL_PATH = './model/fc_model.pth'
TRAIN_SIZE = 1500
TEST_SIZE = 200

# load data
df = pd.read_csv(DATA_PATH)

def parse_array(col):
    return np.array(eval(col))  

r = np.stack(df['r'].apply(parse_array))
alpha = np.stack(df['alpha'].apply(parse_array))
beta = np.stack(df['beta'].apply(parse_array))
x = np.stack(df['x'].apply(parse_array))

# test data
test_conditions = torch.tensor(np.hstack([r[TRAIN_SIZE:], alpha[TRAIN_SIZE:], beta[TRAIN_SIZE:]]), dtype=torch.float32)
test_targets = torch.tensor(x[TRAIN_SIZE:], dtype=torch.float32)

# model - fully connect
class FCModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=128):
        super(FCModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim)
        )

    def forward(self, condition):
        return self.model(condition)

# load model
input_dim = test_targets.shape[1]
condition_dim = test_conditions.shape[1]
model = FCModel(input_dim=input_dim, condition_dim=condition_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# test
with torch.no_grad():
    predictions = model(test_conditions)

# check results
success_count = 0
B = 100
for i in range(TEST_SIZE):
    if test_targets[i].numpy().sum() > B + 1e-5:
        print(test_targets[i].numpy().sum())
        print("Generated data were wrong.")
        break
    
    pro_actual = np.dot(test_targets[i].numpy(), r[TRAIN_SIZE+i])
    pro_predicted = np.dot(predictions[i].numpy(), r[TRAIN_SIZE+i])
    print(f"Test {i+1}:")
    print(f"  Predicted:      {predictions[i].numpy()}")
    print(f"  Actual:         {test_targets[i].numpy()}")
    print(f"  Minimum:        {alpha[TRAIN_SIZE+i] * B}")
    print(f"  Maximum:        {beta[TRAIN_SIZE+i] * B}")
    print(f"  rate:           {r[TRAIN_SIZE+i]}")
    # Profit
    print(f"  Predicted Profit:                 {pro_predicted}")
    print(f"  Actual Profit:                    {pro_actual}")
    print(f"  Profit diff (Actual - Predicted): {pro_actual - pro_predicted}")
    
    try:
        assert predictions[i].numpy().sum() <= B + 1e-5, "Budget constraint violated"
        assert np.all(predictions[i].numpy() >= alpha[TRAIN_SIZE+i] * B), "Minimum investment ratio violated"
        assert np.all(predictions[i].numpy() <= beta[TRAIN_SIZE+i] * B), "Maximum investment ratio violated"
        success_count += 1
    except AssertionError as e:
        print(f"Test {i + 1} failed: {e}")

success_ratio = success_count / TEST_SIZE
print(f"Success Ratio for fully_connect model: {success_ratio * 100:.2f}%")