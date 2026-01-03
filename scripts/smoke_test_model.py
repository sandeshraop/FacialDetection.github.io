import yaml
import torch
import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader.casme_loader import get_data_loaders
from src.models.hybrid_model import build_model

def main():
    with open('config/config.yaml','r') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['batch_size'] = 2
    cfg['data']['num_workers'] = 0
    cfg['data']['use_optical_flow'] = False  # speed up test
    train_loader, _, _ = get_data_loaders(cfg['data'])
    model = build_model(cfg['model'])
    model.eval()
    batch = next(iter(train_loader))
    inputs = batch['frames']
    print('Input shape:', inputs.shape)
    with torch.no_grad():
        outputs = model(inputs)
    print('Output shape:', outputs.shape)
    print('OK')

if __name__ == '__main__':
    main()
