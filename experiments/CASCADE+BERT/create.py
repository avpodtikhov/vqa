import json
import torch
from config import *

with open(model_log, 'w') as f:
    json.dump({'acc': 0, 'epoch': 0, 'loss': 0}, f)
print('Log data saved succesfully')

state = {
    'state_dict': 0,
    'optimizer': 0,
    'lr_base': 0
}
torch.save(state, model_pth)
print('Model saved sucessfully')