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
    att_texts = []
    def hook_fn2(m, i, o):
        att_texts.append(list(torch.nn.functional.softmax(o, dim=1).view(14).cpu().data.numpy()))
        return
    dataset = VQAv2()
    ds = Subset(dataset, range(dataset.data_size // 10, dataset.data_size))
    ds_val = Subset(dataset, range(dataset.data_size // 10))

    data_size = dataset.data_size
    ans_size = dataset.ans_size

    model = MCAN(ans_size)
    model.attflat_img.mlp[3].register_forward_hook(hook_fn)
    model.attflat_lang.mlp[3].register_forward_hook(hook_fn1)
    model.attflat_text.mlp[3].register_forward_hook(hook_fn2)
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
        steps = []
        for step, (x_batch, y_batch, z_batch, idx, cat, texts_feat, texts) in tqdm(enumerate(dataloader)):
            img_feat_iter = x_batch.cuda()
            ques_ix_iter = y_batch.cuda()
            ans_iter = z_batch.cuda()
            texts_feat = texts_feat.cuda()

            pred = model(img_feat_iter, ques_ix_iter, texts_feat)            # pred = pred.masked_fill(cat.cuda(), 0.)
            pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
            ans = []
            for i in range(1):
                ans_ = '???'
                if pred_argmax[i] >= len(dataset.ans_to_ix):
                    ans_ = texts[pred_argmax[i] - len(dataset.ans_to_ix)]
                    print(step, ans_)
                ans.append(ans_)
            answer = ans[0]
            ans = dataset.ans_list[idx]
            qid = int(ans['question_id'])
            qid_list.append(qid)
            ans_list.append(answer)
            steps.append(step)
            if step > 10000:
                break
        d = {'QusstionID': qid_list, 'Answer': ans_list, 'AttIMG' : att_img, 'AttQNS1' : att_qns[::2], 'AttQNS2' : att_qns[1::2], 'AttT' : att_texts}
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
