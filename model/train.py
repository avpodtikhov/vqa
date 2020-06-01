from configs.parser import get_cfg
from final.load_data import VQAv2
from final.optim import adjust_lr, get_optim
from final.net import MCAN
import torch.nn as nn
from tqdm import tqdm
from final.proc import train, test
import json
import time
import torch
from final.proc import test, train
import os

with torch.cuda.device(3):
    cfg = get_cfg()
    ds = VQAv2(cfg, 'train')
    ds_val = VQAv2(cfg, 'val')
    data_size = ds.data_size
    ans_size = ds.ans_size

    model = MCAN(cfg, ans_size)
    model.cuda()
    model.train()

    resume = True

    if resume and os.path.exists(cfg['model_log']):
        with open(cfg['model_log'], "r") as fp:
            log = json.load(fp)
        start_epoch = int(log['epoch']) 
        print('Find model in epoch', log['epoch'], ' | ', 'Accuracy', log['acc'], ' | ', 'Loss', log['loss'])
        params = torch.load(cfg['model_pth'])
        model.load_state_dict(params['state_dict'])
        optim = get_optim(cfg, model, data_size, params['lr_base'])
        optim._step = int(data_size / 64 * start_epoch)
        optim.optimizer.load_state_dict(params['optimizer'])
    else:
        start_epoch = 0
        optim = get_optim(cfg, model, data_size)

    for epoch in range(start_epoch, cfg['MAX_EPOCH']):
        print('Epoch #', epoch + 1)
        if epoch in cfg['LR_DECAY_LIST']:
            adjust_lr(optim, cfg['LR_DECAY_R'])

        time_start = time.time()
        train_loss, train_acc = train(ds, optim, model)
        time_end = time.time()

        print('Epoch #', epoch + 1, '| Train Finished in {}s'.format(int(time_end-time_start)))
        print('Loss: ', train_loss, 'Accuracy: ', train_acc)
        print('')

        with open(cfg['model_log'], 'w') as f:
            json.dump({'acc': train_acc, 'epoch': epoch + 1, 'loss': train_loss}, f)
        print('Log data saved succesfully')

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base
        }
        torch.save(state, cfg['model_pth'])
        print('Model saved sucessfully')
        time_start = time.time()
        val_loss, val_acc, losses = test(ds_val, model)
        time_end = time.time()

        print('Validation Finished in {}s'.format(int(time_end-time_start)))
        print('Loss: ', val_loss, 'Accuracy: ', val_acc)
        print('')
        print('-------------')

