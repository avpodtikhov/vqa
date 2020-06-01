from torch.utils.data import DataLoader
import torch
import numpy as np
from final.preprocessing import prep_ans
loss_fn = torch.nn.BCELoss(reduction='sum').cuda()


def train(ds, optimizer, model, batchsize=64):
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
    losses = []
    for step, (x_batch, y_batch, z_batch, idx, cat, texts_feat, texts, box) in enumerate(dataloader):
        optimizer.zero_grad()
        img_feat_iter = x_batch.cuda()
        ques_ix_iter = y_batch.cuda()
        ans_iter = z_batch.cuda()
        texts_feat = texts_feat.cuda()
        box = box.cuda()

        pred = model(img_feat_iter, ques_ix_iter, texts_feat, box)
        pred = pred.masked_fill(cat.cuda(), 0.)

        loss = loss_fn(pred, ans_iter)
        loss.backward()
        loss_sum += loss.cpu().data.numpy()
        optimizer.step()

        pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
        ans = []
        for i in range(batchsize):
            if pred_argmax[i] >= len(ds.ans_to_ix):
                ans_ = texts[pred_argmax[i] - len(ds.ans_to_ix)]
            else:
                ans_ = ds.ix_to_ans[str(pred_argmax[i])]
            ans.append(ans_)
        # print(pred_argmax.shape[0], idx)
        real = [[ds.ans_list[idx[i].item()]['answers'][j]['answer'] for j in range(len(ds.ans_list[idx[i].item()]['answers']))] for i in range(batchsize)]
        acc = [min(1, np.sum([(ans[i]== prep_ans(answer)) for answer in real[i]]) / 3) for i in range(batchsize)]
        gtc_acc += (float(np.sum(acc)) / batchsize)
        if step % 340 == 0 and step != 0:
            losses.append(loss.cpu().data.numpy())
            print('->Loss: ', loss_sum /  step)
            print('->Accuracy: ', gtc_acc /  step)
            print('->Time (800 batchs): ', step, '/', (592300 // batchsize // 340) * 340)
            print('____')
    return loss_sum / step, gtc_acc / step

def test(ds_val, model, batchsize=1):
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
    losses = []
    for step, (x_batch, y_batch, z_batch, idx, cat, texts_feat, texts, box) in enumerate(dataloader):
        img_feat_iter = x_batch.cuda()
        ques_ix_iter = y_batch.cuda()
        ans_iter = z_batch.cuda()
        box = box.cuda()

        texts_feat = texts_feat.cuda()

        pred = model(img_feat_iter, ques_ix_iter, texts_feat, box)
        pred = pred.masked_fill(cat.cuda(), 0.)
        loss = loss_fn(pred, ans_iter)
        loss_sum += loss.cpu().data.numpy()

        pred_argmax = np.argmax(pred.cpu().data.numpy(), axis=1)
        ans = []
        for i in range(batchsize):
            if pred_argmax[i] >= len(ds_val.ans_to_ix):
                ans_ = texts[pred_argmax[i] - len(ds_val.ans_to_ix)]
            else:
                ans_ = ds_val.ix_to_ans[str(pred_argmax[i])]
            ans.append(ans_)
        # print(pred_argmax.shape[0], idx)
        real = [[ds_val.ans_list[idx[i].item()]['answers'][j]['answer'] for j in range(len(ds_val.ans_list[idx[i].item()]['answers']))] for i in range(batchsize)]
        acc = [min(1, np.sum([(ans[i]== prep_ans(answer)) for answer in real[i]]) / 3) for i in range(batchsize)]
        gtc_acc += (float(np.sum(acc)) / batchsize)
        if step % 10000 == 0 and step != 0:
            losses.append(loss.cpu().data.numpy())
            print('->Loss: ', loss_sum /  step)
            print('->Accuracy: ', gtc_acc /  step)
            print('->Time (10000 batchs): ', step, '/', (65811 // batchsize // 8000) * 8000)
            print('____')
            return loss_sum / (step + 1),  gtc_acc / (step + 1), losses
    return loss_sum / (step + 1),  gtc_acc / (step + 1), losses