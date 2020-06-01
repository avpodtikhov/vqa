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
import pandas as pd
from tqdm import tqdm

with torch.cuda.device(1):
    att_img = []
    def hook_fn(m, i, o):
        att_img.append(list(torch.nn.functional.softmax(o, dim=1).view(100).cpu().data.numpy()))
        return
    att_qns = []
    def hook_fn1(m, i, o):
        att_qns.append(list(torch.nn.functional.softmax(o, dim=1).view(36).cpu().data.numpy()))
        return
    dataset = VQAv2()
    ds = Subset(dataset, range(dataset.data_size // 10, dataset.data_size))
    ds_val = Subset(dataset, range(dataset.data_size // 10))

    data_size = dataset.data_size
    ans_size = dataset.ans_size

    model = MCAN(ans_size)
    model.attflat_img.mlp[3].register_forward_hook(hook_fn)
    model.attflat_lang.mlp[3].register_forward_hook(hook_fn1)
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
        ans_list = []
        qid_list = []
        for step, (x_batch, y_batch, z_batch, b_batch, idx) in tqdm(enumerate(dataloader)):
            img_feat_iter = x_batch.cuda()
            ques_ix_iter = y_batch.cuda()
            ans_iter = z_batch.cuda()
            box_iter = b_batch.cuda()

            pred = model(img_feat_iter, ques_ix_iter, box_iter)
            pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
            
            ans = dataset.ans_list[idx]
            qid = int(ans['question_id'])
            answer = dataset.ix_to_ans[str(pred_argmax[0])]
            qid_list.append(qid)
            ans_list.append(answer)
            if step > 1000:
                break
        d = {'QusstionID': qid_list, 'Answer': ans_list, 'AttIMG' : att_img, 'AttQNS' : att_qns}
        df = pd.DataFrame(data=d)
        df.to_csv('ans1.csv', index=False)

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
    test(model)
