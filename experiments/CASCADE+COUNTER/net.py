## Model architecture

import math
import torch.nn as nn
import torch.nn.functional as F
from config import *
import torch
from counting import Counter

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MHAtt(nn.Module):
    def __init__(self):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            multi_head,
            hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            multi_head,
            hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            multi_head,
            hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = nn.Sequential(nn.Linear(hidden_size, fc_size), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(dropout_rate), 
                                 nn.Linear(fc_size, hidden_size))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = LayerNorm(hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()

        self.mhatt = MHAtt()
        self.ffn = nn.Sequential(nn.Linear(hidden_size, fc_size), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(dropout_rate), 
                                 nn.Linear(fc_size, hidden_size))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x

class MCA_ED(nn.Module):
    def __init__(self):
        super(MCA_ED, self).__init__()
        
        self.enc_list = nn.ModuleList([SA() for _ in range(num_layers)])
        self.dec_list = nn.ModuleList([SGA() for _ in range(num_layers)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

class AttFlat(nn.Module):
    def __init__(self):
        super(AttFlat, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(hidden_size, flat_mlp_size), 
                                 nn.ReLU(inplace=True), 
                                 nn.Dropout(dropout_rate), 
                                 nn.Linear(flat_mlp_size, 1))

        self.linear_merge = nn.Linear(
            hidden_size,
            flat_out_size
        )
        self.count = Counter(20)
        self.c_lin = nn.Linear(21, flat_out_size)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(flat_out_size)

    def forward(self, x, x_mask, box=None):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        # print('att : ', att.size())
            # count1 = att.view(-1, 36, 1)
        att1 = F.softmax(att, dim=1)
        # print('att + softmax: ', att.size())
        
        att_list = []
        for i in range(1):
            att_list.append(
                torch.sum(att1[:, :, i: i + 1] * x, dim=1)
            )
        # print('att_list : ', len(att_list), att_list[0].size())

        x_atted = torch.cat(att_list, dim=1)
        # print('x_atted : ', x_atted.size())

        x_atted = self.linear_merge(x_atted)
        if box is not None:
            att = att.squeeze(2)
            count1 = self.count(box, att)
            x_count = self.bn2(self.relu(self.c_lin(count1)))
            x_atted += x_count
        # print('x_atted(2) : ', x_atted.size())

        return x_atted

class MCAN(nn.Module):
    def __init__(self, answer_size):
        super(MCAN, self).__init__()
        '''
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=glove_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.lstm = nn.LSTM(
            input_size=glove_size,
            hidden_size=hidden_size
        )
        '''
        self.ques_feat_linear  = nn.Linear(
            768,
            hidden_size
        )

        self.img_feat_linear = nn.Linear(
            img_feat_size,
            hidden_size
        )

        self.backbone = MCA_ED()

        self.attflat_img = AttFlat()
        self.attflat_lang = AttFlat()

        self.proj_norm = LayerNorm(flat_out_size)
        self.proj = nn.Linear(flat_out_size, answer_size)


    def forward(self, img_feat, ques_ix, box):
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix)
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        '''
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat
        '''
        lang_feat = self.ques_feat_linear(ques_ix)
        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask,
            box
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)