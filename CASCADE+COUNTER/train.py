import config as my_cfg
from load_data import VQAv2
from optim import *
from net import MCAN
from torch.utils.data import Subset
import time
from tqdm import tqdm
from config import *
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from preproccessing import *
import torch

with torch.cuda.device(3):
    dataset = VQAv2()
    ds = Subset(dataset, range(dataset.data_size // 10, dataset.data_size))
    ds_val = Subset(dataset, range(dataset.data_size // 10))

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    model = MCAN(ans_size)
    model.cuda()
    model.train()

    loss_fn = nn.BCELoss(reduction='sum').cuda()

    def train(optimizer, model, batchsize=64):
        model.train()
        dataloader = DataLoader(
                    ds,
                    batch_size=batchsize,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True)
        loss_sum = 0
        gtc_acc = 0.
        time_start = time.time()
        losses = []
        for step, (x_batch, y_batch, z_batch, b_batch, idx) in enumerate(dataloader):
            optimizer.zero_grad()
            img_feat_iter = x_batch.cuda()
            ques_ix_iter = y_batch.cuda()
            ans_iter = z_batch.cuda()
            box_iter = b_batch.cuda()

            pred = model(img_feat_iter, ques_ix_iter, box_iter)

            loss = loss_fn(pred, ans_iter)
            loss.backward()
            loss_sum += loss.cpu().data.numpy()
            optim.step()

            pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
            # print(pred_argmax.shape[0], idx)
            real = [[dataset.ans_list[idx[i].item()]['answers'][j]['answer'] for j in range(len(dataset.ans_list[idx[i].item()]['answers']))] for i in range(batchsize)]
            acc = [min(1, np.sum([(dataset.ix_to_ans[str(pred_argmax[i])] == prep_ans(answer)) for answer in real[i]]) / 3) for i in range(batchsize)]
            gtc_acc += (float(np.sum(acc)) / batchsize)
            if step % 340 == 0 and step != 0:
                losses.append(loss.cpu().data.numpy())
                time_end = time.time()
                t = int(time_end-time_start)
                print('->Loss: ', loss_sum /  step)
                print('->Accuracy: ', gtc_acc /  step)
                print('->Time (800 batchs): ', step, '/', (592300 // batchsize // 340) * 340)
                print('____')
                time_start = time.time()

        return loss_sum / step, gtc_acc / step

    def test(model, batchsize=1):
        model.eval()
        dataloader = DataLoader(
                    ds_val,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True)
        loss_sum = 0
        gtc_acc = 0
        time_start = time.time()
        losses = []
        for step, (x_batch, y_batch, z_batch, b_batch, idx) in enumerate(dataloader):
            img_feat_iter = x_batch.cuda()
            ques_ix_iter = y_batch.cuda()
            ans_iter = z_batch.cuda()
            box_iter = b_batch.cuda()

            pred = model(img_feat_iter, ques_ix_iter, box_iter)
            loss = loss_fn(pred, ans_iter)
            loss_sum += loss.cpu().data.numpy()

            pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
            # print(pred_argmax.shape[0], idx)
            real = [[dataset.ans_list[idx[i].item()]['answers'][j]['answer'] for j in range(len(dataset.ans_list[idx[i].item()]['answers']))] for i in range(batchsize)]
            acc = [min(1, np.sum([(dataset.ix_to_ans[str(pred_argmax[i])] == prep_ans(answer)) for answer in real[i]]) / 3) for i in range(batchsize)]
            gtc_acc += (float(np.sum(acc)) / batchsize)
            if step % 10000 == 0 and step != 0:
                losses.append(loss.cpu().data.numpy())
                time_end = time.time()
                t = int(time_end-time_start)
                print('->Loss: ', loss_sum /  step)
                print('->Accuracy: ', gtc_acc /  step)
                print('->Time (10000 batchs): ', step, '/', (65811 // batchsize // 8000) * 8000)
                print('____')
                return loss_sum / (step + 1),  gtc_acc / (step + 1), losses
                time_start = time.time()
        return loss_sum / (step + 1),  gtc_acc / (step + 1), losses

    from tqdm import tqdm
    import time

    # Load model from checkpoint
    import json

    resume = True

    with open(model_log, "r") as fp:
        log = json.load(fp) 

    if resume and int(log['epoch']):
        start_epoch = int(log['epoch']) 
        print('Find model in epoch', log['epoch'], ' | ', 'Accuracy', log['acc'], ' | ', 'Loss', log['loss'])
        params = torch.load(model_pth)
        model.load_state_dict(params['state_dict'])
        optim = get_optim(model, data_size, params['lr_base'])
        optim._step = int(data_size / 64 * start_epoch)
        optim.optimizer.load_state_dict(params['optimizer'])
    else:
        start_epoch = 0
        optim = get_optim(model, data_size)

    for epoch in range(start_epoch, MAX_EPOCH):
        print('Epoch #', epoch + 1)

        if epoch in LR_DECAY_LIST:
            adjust_lr(optim, LR_DECAY_R)

        time_start = time.time()
        train_loss, train_acc = train(optim, model)
        time_end = time.time()

        print('Epoch #', epoch + 1, '| Train Finished in {}s'.format(int(time_end-time_start)))
        print('Loss: ', train_loss, 'Accuracy: ', train_acc)
        print('')

        with open(model_log, 'w') as f:
            json.dump({'acc': train_acc, 'epoch': epoch + 1, 'loss': train_loss}, f)
        print('Log data saved succesfully')

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optim.optimizer.state_dict(),
            'lr_base': optim.lr_base
        }
        torch.save(state, model_pth)
        print('Model saved sucessfully')
        time_start = time.time()
        val_loss, val_acc, losses = test(model)
        time_end = time.time()

        print('Validation Finished in {}s'.format(int(time_end-time_start)))
        print('Loss: ', val_loss, 'Accuracy: ', val_acc)
        print('')
        print('-------------')