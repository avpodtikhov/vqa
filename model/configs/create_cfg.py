import yaml

# Paths
cfg = {'IMAGE_MODEL' : 'PYTHIA',
       'QUES MODEL' : 'ALBERT',
       'COUNTER' : False,
       'CLASS' : False,
       'model_log' : '/mnt/data/users/apodtikhov/vqa/models/pythia/log.json',
       'model_pth' :  '/mnt/data/users/apodtikhov/vqa/models/pythia/model.pth',
       'img_feat_pad_size' : 100,
       'max_token' : 36,
       'hidden_size' : 512,
       'word_embedding_size' : 4096,
       'img_feat_size' : 2048,
       'num_layers' : 6,
       'dropout_rate' : 0.1,
       'fc_size' : 4 * 512,
       'multi_head' : 8,
       'hidden_size_head' : int(512 / 8),
       'flat_out_size' : 1024,
       'flat_mlp_size' : 512,
       'LR_BASE' : 0.0001,
       'OPT_BETAS' : (0.9, 0.98),
       'OPT_EPS' : 1e-9,
       'BATCH_SIZE' : 64,
       'MAX_EPOCH' : 13,
       'LR_DECAY_LIST' : [10, 12],
       'LR_DECAY_R' : 0.2,
}

import argparse
import pickle

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-IMG_FEAT', dest='img_feat', choices=['bottom-up', 'panoptic', 'mask', 'pythia'], help='Object detection model', type=str, required=True)
    parser.add_argument('-QUES_FEAT', dest='ques_feat', choices=['gru', 'bert', 'albert', 'roberta'], help='Question embedding model', type=str, required=True)
    parser.add_argument('-COUNTER', action='store_true',dest='cnt', help='Counter module')
    parser.add_argument('-CLASS', action='store_true',dest='classification', help='Add classification')
    return parser.parse_args()

args = argument_parser()
cfg_file = './cfg_' + args.img_feat + '_'  + args.ques_feat + '_cnt'  * args.cnt + '_class'  * args.classification + '.yaml'

import yaml
with open(cfg_file, 'w') as f:
    yaml.dump(cfg, f)
