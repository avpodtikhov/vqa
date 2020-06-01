import argparse
import yaml

def get_cfg():
    cfg_file = './configs/cfg_final.yaml'
    with open(cfg_file, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg