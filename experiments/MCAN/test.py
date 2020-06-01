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

    model = MCAN(pretrained_emb, token_size, ans_size)
    model.cuda()
    model.eval()

    loss_fn = nn.BCELoss(reduction='sum').cuda()

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
        by_question_type = {}
        by_answer_type = {}
        for step, (x_batch, y_batch, z_batch, idx) in tqdm(enumerate(dataloader)):
            img_feat_iter = x_batch.cuda()
            ques_ix_iter = y_batch.cuda()
            ans_iter = z_batch.cuda()

            pred = model(img_feat_iter, ques_ix_iter)

            pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
            real = [[dataset.ans_list[idx[i].item()]['answers'][j]['answer'] for j in range(len(dataset.ans_list[idx[i].item()]['answers']))] for i in range(batchsize)]
            acc = [min(1, np.sum([(dataset.ix_to_ans[str(pred_argmax[i])] == prep_ans(answer)) for answer in real[i]]) / 3) for i in range(batchsize)]
            accuracy = acc[0]
            # print(pred_argmax.shape[0], idx)
            ans_stat = dataset.ans_list[idx]
            qt = ans_stat['question_type']
            if by_question_type.get(qt, -1) == -1:
                by_question_type[qt] = [0., 0]
            at = ans_stat['answer_type']
            if by_answer_type.get(at, -1) == -1:
                by_answer_type[at] = [0., 0]
            by_answer_type[at][0] += accuracy
            by_answer_type[at][1] += 1
            by_question_type[qt][0] += accuracy
            by_question_type[qt][1] += 1
        return by_question_type, by_answer_type
    from tqdm import tqdm
    import time
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
    time_start = time.time()
    by_question_type, by_answer_type = test(model)
    print(by_question_type, by_answer_type)
    json.dump(by_question_type, open( "by_question_type.json", 'w'))
    json.dump(by_answer_type, open( "by_answer_type.json", 'w'))
    
    time_end = time.time()
    print('Validation Finished in {}s'.format(int(time_end-time_start)))